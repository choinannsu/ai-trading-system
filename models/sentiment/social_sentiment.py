"""
Social Media Sentiment Analysis
Reddit, Twitter, and community sentiment analysis with bot filtering
"""

import numpy as np
import pandas as pd
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
import asyncio
import aiohttp
import requests
from collections import defaultdict, Counter
import time

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import praw  # Reddit API
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class SocialConfig:
    """Configuration for social media sentiment analysis"""
    # Platform settings
    platforms: List[str] = None
    reddit_client_id: str = None
    reddit_client_secret: str = None
    reddit_user_agent: str = "financial_sentiment_bot"
    twitter_bearer_token: str = None
    
    # Analysis settings
    min_followers: int = 10
    min_karma: int = 100
    max_posts_per_hour: int = 1000
    sentiment_threshold: float = 0.1
    
    # Bot detection
    enable_bot_filtering: bool = True
    bot_detection_threshold: float = 0.7
    min_account_age_days: int = 30
    
    # Community weights
    community_weights: Dict[str, float] = None
    
    # Keywords and tickers
    stock_tickers: List[str] = None
    financial_keywords: List[str] = None
    
    # Time settings
    lookback_hours: int = 24
    time_decay_factor: float = 0.1
    
    def __post_init__(self):
        if self.platforms is None:
            self.platforms = ['reddit', 'twitter']
        
        if self.community_weights is None:
            self.community_weights = {
                'wallstreetbets': 1.5,
                'investing': 1.2,
                'stocks': 1.0,
                'SecurityAnalysis': 1.3,
                'financialindependence': 0.8,
                'personalfinance': 0.6,
                'cryptocurrency': 1.1,
                'default': 0.5
            }
        
        if self.stock_tickers is None:
            self.stock_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
                'SPY', 'QQQ', 'VTI', 'BTC', 'ETH'
            ]
        
        if self.financial_keywords is None:
            self.financial_keywords = [
                'bull', 'bear', 'moon', 'crash', 'dip', 'rally', 'squeeze',
                'diamond hands', 'paper hands', 'hodl', 'buy the dip',
                'to the moon', 'rocket', 'tendies', 'yolo'
            ]


@dataclass
class SocialPost:
    """Individual social media post"""
    id: str
    platform: str
    author: str
    content: str
    timestamp: datetime
    upvotes: int = 0
    downvotes: int = 0
    comments: int = 0
    shares: int = 0
    community: str = None
    url: str = None
    is_bot: bool = False
    author_metrics: Dict[str, Any] = None


@dataclass
class AuthorProfile:
    """Social media author profile for bot detection"""
    username: str
    account_age_days: int
    follower_count: int = 0
    following_count: int = 0
    post_count: int = 0
    karma: int = 0
    verified: bool = False
    posting_frequency: float = 0.0  # posts per day
    avg_post_length: float = 0.0
    duplicate_content_ratio: float = 0.0


@dataclass
class SentimentIndex:
    """Real-time sentiment index"""
    timestamp: datetime
    overall_sentiment: float
    sentiment_distribution: Dict[str, float]
    volume_weighted_sentiment: float
    community_breakdown: Dict[str, float]
    platform_breakdown: Dict[str, float]
    top_tickers: List[Tuple[str, float]]
    trending_topics: List[str]
    bot_ratio: float
    confidence_score: float
    total_posts: int


class BotDetector:
    """Detect bots and spam accounts in social media"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        
    def analyze_author(self, profile: AuthorProfile, recent_posts: List[SocialPost]) -> float:
        """Calculate bot probability for an author"""
        bot_score = 0.0
        
        # Account age factor (very new accounts are suspicious)
        if profile.account_age_days < 7:
            bot_score += 0.3
        elif profile.account_age_days < 30:
            bot_score += 0.1
        
        # Posting frequency (too high or too regular is suspicious)
        if profile.posting_frequency > 50:  # More than 50 posts per day
            bot_score += 0.3
        elif profile.posting_frequency > 20:
            bot_score += 0.1
        
        # Follower/following ratio
        if profile.follower_count > 0 and profile.following_count > 0:
            ratio = profile.following_count / profile.follower_count
            if ratio > 10:  # Following way more than followers
                bot_score += 0.2
        
        # Content analysis
        if recent_posts:
            # Check for duplicate or very similar content
            content_similarity = self._calculate_content_similarity(recent_posts)
            bot_score += content_similarity * 0.3
            
            # Check for very short or very long posts consistently
            post_lengths = [len(post.content) for post in recent_posts]
            if post_lengths:
                avg_length = np.mean(post_lengths)
                std_length = np.std(post_lengths)
                
                # Very consistent length might indicate automation
                if std_length < 10 and len(recent_posts) > 5:
                    bot_score += 0.2
                
                # Very short posts consistently
                if avg_length < 20:
                    bot_score += 0.1
        
        # Karma/post ratio (low engagement might indicate bot)
        if profile.post_count > 0:
            karma_per_post = profile.karma / profile.post_count
            if karma_per_post < 0.1:  # Very low engagement
                bot_score += 0.1
        
        return min(1.0, bot_score)
    
    def _calculate_content_similarity(self, posts: List[SocialPost]) -> float:
        """Calculate content similarity ratio"""
        if len(posts) < 2:
            return 0.0
        
        contents = [post.content.lower() for post in posts]
        total_pairs = 0
        similar_pairs = 0
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                total_pairs += 1
                
                # Simple similarity check
                content1_words = set(contents[i].split())
                content2_words = set(contents[j].split())
                
                if len(content1_words) > 0 and len(content2_words) > 0:
                    intersection = len(content1_words & content2_words)
                    union = len(content1_words | content2_words)
                    similarity = intersection / union
                    
                    if similarity > 0.7:  # 70% similar
                        similar_pairs += 1
        
        return similar_pairs / max(total_pairs, 1)
    
    def is_bot(self, profile: AuthorProfile, recent_posts: List[SocialPost]) -> bool:
        """Determine if author is likely a bot"""
        bot_probability = self.analyze_author(profile, recent_posts)
        return bot_probability >= self.threshold


class RedditCollector:
    """Collect sentiment data from Reddit"""
    
    def __init__(self, config: SocialConfig):
        self.config = config
        self.reddit = None
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit API client"""
        if not PRAW_AVAILABLE:
            logger.warning("PRAW not available. Install with: pip install praw")
            return
        
        if not all([self.config.reddit_client_id, self.config.reddit_client_secret]):
            logger.warning("Reddit API credentials not provided")
            return
        
        try:
            self.reddit = praw.Reddit(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent=self.config.reddit_user_agent
            )
            logger.info("Reddit API initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
    
    def collect_posts(self, subreddit_names: List[str], limit: int = 100) -> List[SocialPost]:
        """Collect posts from specified subreddits"""
        if not self.reddit:
            return []
        
        posts = []
        
        for subreddit_name in subreddit_names:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for submission in subreddit.hot(limit=limit):
                    # Filter by keywords or tickers
                    if self._contains_financial_content(submission.title + ' ' + submission.selftext):
                        post = SocialPost(
                            id=submission.id,
                            platform='reddit',
                            author=str(submission.author) if submission.author else 'deleted',
                            content=f"{submission.title}. {submission.selftext}",
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            upvotes=submission.score,
                            downvotes=0,  # Not directly available
                            comments=submission.num_comments,
                            community=subreddit_name,
                            url=f"https://reddit.com{submission.permalink}"
                        )
                        posts.append(post)
                
                # Also get comments from top posts
                for submission in subreddit.hot(limit=10):
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:20]:
                        if self._contains_financial_content(comment.body):
                            post = SocialPost(
                                id=comment.id,
                                platform='reddit',
                                author=str(comment.author) if comment.author else 'deleted',
                                content=comment.body,
                                timestamp=datetime.fromtimestamp(comment.created_utc),
                                upvotes=comment.score,
                                comments=0,
                                community=subreddit_name,
                                url=f"https://reddit.com{comment.permalink}"
                            )
                            posts.append(post)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                continue
        
        return posts
    
    def _contains_financial_content(self, text: str) -> bool:
        """Check if text contains financial keywords or tickers"""
        text_upper = text.upper()
        text_lower = text.lower()
        
        # Check for stock tickers
        for ticker in self.config.stock_tickers:
            if f"${ticker}" in text_upper or f" {ticker} " in text_upper:
                return True
        
        # Check for financial keywords
        for keyword in self.config.financial_keywords:
            if keyword.lower() in text_lower:
                return True
        
        return False


class TwitterCollector:
    """Collect sentiment data from Twitter (placeholder)"""
    
    def __init__(self, config: SocialConfig):
        self.config = config
    
    def collect_tweets(self, keywords: List[str], limit: int = 100) -> List[SocialPost]:
        """Collect tweets (placeholder implementation)"""
        # Note: Twitter API v2 requires different authentication
        # This is a placeholder for demonstration
        logger.warning("Twitter collection not implemented - requires Twitter API v2 setup")
        return []


class SocialSentimentAnalyzer:
    """Analyze sentiment from social media platforms"""
    
    def __init__(self, config: SocialConfig = None):
        self.config = config or SocialConfig()
        self.bot_detector = BotDetector(self.config.bot_detection_threshold)
        
        # Initialize collectors
        self.reddit_collector = RedditCollector(self.config)
        self.twitter_collector = TwitterCollector(self.config)
        
        # Cache for author profiles
        self.author_cache = {}
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of text using TextBlob or rule-based approach"""
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                confidence = abs(polarity)
                return polarity, confidence
            except:
                pass
        
        # Rule-based sentiment analysis
        positive_words = [
            'moon', 'rocket', 'bull', 'up', 'gain', 'profit', 'buy', 'long',
            'diamond', 'hodl', 'rally', 'pump', 'squeeze', 'tendies'
        ]
        
        negative_words = [
            'bear', 'crash', 'dump', 'down', 'loss', 'sell', 'short',
            'paper', 'drop', 'fall', 'decline', 'rip', 'dead'
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0, 0.0
        
        sentiment = (pos_count - neg_count) / total
        confidence = min(1.0, total / 10)  # Higher word count = higher confidence
        
        return sentiment, confidence
    
    def _get_author_profile(self, post: SocialPost) -> AuthorProfile:
        """Get or create author profile"""
        if post.author in self.author_cache:
            return self.author_cache[post.author]
        
        # Create basic profile (in real implementation, this would fetch from API)
        profile = AuthorProfile(
            username=post.author,
            account_age_days=365,  # Default assumption
            karma=self.config.min_karma,
            post_count=100
        )
        
        self.author_cache[post.author] = profile
        return profile
    
    def _filter_bots(self, posts: List[SocialPost]) -> List[SocialPost]:
        """Filter out bot posts if bot detection is enabled"""
        if not self.config.enable_bot_filtering:
            return posts
        
        filtered_posts = []
        author_posts = defaultdict(list)
        
        # Group posts by author
        for post in posts:
            author_posts[post.author].append(post)
        
        # Analyze each author
        for author, author_post_list in author_posts.items():
            profile = self._get_author_profile(author_post_list[0])
            
            if not self.bot_detector.is_bot(profile, author_post_list):
                filtered_posts.extend(author_post_list)
            else:
                # Mark as bot
                for post in author_post_list:
                    post.is_bot = True
        
        return filtered_posts
    
    def _calculate_post_weight(self, post: SocialPost) -> float:
        """Calculate weight for a post based on various factors"""
        weight = 1.0
        
        # Community weight
        community_weight = self.config.community_weights.get(
            post.community, self.config.community_weights['default']
        )
        weight *= community_weight
        
        # Engagement weight (upvotes, comments)
        engagement_score = post.upvotes + post.comments * 2
        engagement_weight = min(2.0, 1.0 + np.log1p(engagement_score) / 10)
        weight *= engagement_weight
        
        # Time decay
        hours_old = (datetime.now() - post.timestamp).total_seconds() / 3600
        time_weight = np.exp(-self.config.time_decay_factor * hours_old)
        weight *= time_weight
        
        # Bot penalty
        if post.is_bot:
            weight *= 0.1
        
        return weight
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        tickers = []
        text_upper = text.upper()
        
        # Look for $TICKER format
        ticker_pattern = r'\$([A-Z]{1,5})\b'
        matches = re.findall(ticker_pattern, text_upper)
        tickers.extend(matches)
        
        # Look for known tickers
        for ticker in self.config.stock_tickers:
            if f" {ticker} " in f" {text_upper} ":
                tickers.append(ticker)
        
        return list(set(tickers))
    
    def analyze_posts(self, posts: List[SocialPost]) -> SentimentIndex:
        """Analyze sentiment from social media posts"""
        if not posts:
            return SentimentIndex(
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                sentiment_distribution={'positive': 0, 'negative': 0, 'neutral': 0},
                volume_weighted_sentiment=0.0,
                community_breakdown={},
                platform_breakdown={},
                top_tickers=[],
                trending_topics=[],
                bot_ratio=0.0,
                confidence_score=0.0,
                total_posts=0
            )
        
        # Filter bots
        filtered_posts = self._filter_bots(posts)
        bot_ratio = 1.0 - (len(filtered_posts) / len(posts))
        
        # Analyze sentiment for each post
        sentiments = []
        weights = []
        communities = defaultdict(list)
        platforms = defaultdict(list)
        all_tickers = []
        all_topics = []
        
        for post in filtered_posts:
            sentiment, confidence = self._analyze_text_sentiment(post.content)
            weight = self._calculate_post_weight(post)
            
            sentiments.append(sentiment)
            weights.append(weight)
            
            communities[post.community or 'unknown'].append(sentiment)
            platforms[post.platform].append(sentiment)
            
            # Extract tickers
            tickers = self._extract_tickers(post.content)
            all_tickers.extend(tickers)
            
            # Extract topics (simplified)
            words = post.content.lower().split()
            topics = [word for word in words if word in self.config.financial_keywords]
            all_topics.extend(topics)
        
        # Calculate overall metrics
        if sentiments:
            overall_sentiment = np.average(sentiments, weights=weights)
            volume_weighted_sentiment = np.average(sentiments, weights=[w * (p.upvotes + 1) for w, p in zip(weights, filtered_posts)])
            confidence_score = np.mean([abs(s) for s in sentiments])
        else:
            overall_sentiment = 0.0
            volume_weighted_sentiment = 0.0
            confidence_score = 0.0
        
        # Sentiment distribution
        positive_count = sum(1 for s in sentiments if s > self.config.sentiment_threshold)
        negative_count = sum(1 for s in sentiments if s < -self.config.sentiment_threshold)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        sentiment_distribution = {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }
        
        # Community breakdown
        community_breakdown = {}
        for community, comm_sentiments in communities.items():
            if comm_sentiments:
                community_breakdown[community] = np.mean(comm_sentiments)
        
        # Platform breakdown
        platform_breakdown = {}
        for platform, plat_sentiments in platforms.items():
            if plat_sentiments:
                platform_breakdown[platform] = np.mean(plat_sentiments)
        
        # Top tickers
        ticker_counts = Counter(all_tickers)
        top_tickers = [(ticker, count) for ticker, count in ticker_counts.most_common(10)]
        
        # Trending topics
        topic_counts = Counter(all_topics)
        trending_topics = [topic for topic, count in topic_counts.most_common(10)]
        
        return SentimentIndex(
            timestamp=datetime.now(),
            overall_sentiment=overall_sentiment,
            sentiment_distribution=sentiment_distribution,
            volume_weighted_sentiment=volume_weighted_sentiment,
            community_breakdown=community_breakdown,
            platform_breakdown=platform_breakdown,
            top_tickers=top_tickers,
            trending_topics=trending_topics,
            bot_ratio=bot_ratio,
            confidence_score=confidence_score,
            total_posts=len(posts)
        )
    
    def collect_and_analyze(self, subreddits: List[str] = None, 
                           limit_per_source: int = 100) -> SentimentIndex:
        """Collect data and analyze sentiment"""
        all_posts = []
        
        # Default subreddits
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis']
        
        # Collect from Reddit
        if 'reddit' in self.config.platforms:
            reddit_posts = self.reddit_collector.collect_posts(subreddits, limit_per_source)
            all_posts.extend(reddit_posts)
            logger.info(f"Collected {len(reddit_posts)} Reddit posts")
        
        # Collect from Twitter (placeholder)
        if 'twitter' in self.config.platforms:
            twitter_posts = self.twitter_collector.collect_tweets(
                self.config.stock_tickers, limit_per_source
            )
            all_posts.extend(twitter_posts)
            logger.info(f"Collected {len(twitter_posts)} Twitter posts")
        
        # Analyze sentiment
        sentiment_index = self.analyze_posts(all_posts)
        
        logger.info(f"Analyzed {sentiment_index.total_posts} total posts")
        logger.info(f"Overall sentiment: {sentiment_index.overall_sentiment:.3f}")
        
        return sentiment_index
    
    def create_sentiment_timeseries(self, sentiment_indices: List[SentimentIndex]) -> pd.DataFrame:
        """Create time series from sentiment indices"""
        data = []
        
        for index in sentiment_indices:
            data.append({
                'timestamp': index.timestamp,
                'overall_sentiment': index.overall_sentiment,
                'volume_weighted_sentiment': index.volume_weighted_sentiment,
                'positive_ratio': index.sentiment_distribution['positive'] / max(1, index.total_posts),
                'negative_ratio': index.sentiment_distribution['negative'] / max(1, index.total_posts),
                'neutral_ratio': index.sentiment_distribution['neutral'] / max(1, index.total_posts),
                'bot_ratio': index.bot_ratio,
                'confidence_score': index.confidence_score,
                'total_posts': index.total_posts
            })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df


# Generate sample social media data for testing
def generate_sample_social_posts(count: int = 20) -> List[SocialPost]:
    """Generate sample social media posts for testing"""
    sample_contents = [
        "AAPL to the moon! 🚀 Just bought more shares",
        "Market crash incoming, time to buy puts",
        "Diamond hands on TSLA, not selling",
        "This market is so volatile, paper hands everywhere",
        "SPY calls printing money today",
        "Bitcoin showing strong bullish momentum",
        "Fed meeting tomorrow, expecting dovish tone",
        "Tech stocks are oversold, good buying opportunity",
        "Oil prices surging, XOM looking good",
        "YOLO on meme stocks, what could go wrong",
        "Inflation data coming out, market nervous",
        "Earnings season starting, ready for volatility",
        "Goldman upgrades NVDA, rocket time",
        "Retail investors panic selling again",
        "Buy the dip strategy always works",
        "Short squeeze happening on GME",
        "Market makers manipulating prices",
        "DCA into index funds, boring but works",
        "Crypto winter is here, hodl strong",
        "Wall Street bonuses coming, expect rally"
    ]
    
    communities = ['wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis', 'personalfinance']
    platforms = ['reddit', 'twitter']
    
    posts = []
    for i in range(count):
        posts.append(SocialPost(
            id=f"post_{i}",
            platform=platforms[i % len(platforms)],
            author=f"user_{i % 10}",
            content=sample_contents[i % len(sample_contents)],
            timestamp=datetime.now() - timedelta(hours=i),
            upvotes=np.random.randint(0, 1000),
            comments=np.random.randint(0, 100),
            community=communities[i % len(communities)]
        ))
    
    return posts


# Factory function
def create_social_analyzer(config: Dict[str, Any] = None) -> SocialSentimentAnalyzer:
    """Create social sentiment analyzer with configuration"""
    if config:
        social_config = SocialConfig(**config)
    else:
        social_config = SocialConfig()
    
    return SocialSentimentAnalyzer(social_config)


# Export classes and functions
__all__ = [
    'SocialSentimentAnalyzer',
    'SocialConfig',
    'SocialPost',
    'AuthorProfile',
    'SentimentIndex',
    'BotDetector',
    'RedditCollector',
    'TwitterCollector',
    'create_social_analyzer',
    'generate_sample_social_posts'
]