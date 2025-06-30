"""
News Data Collector for financial news from multiple sources
Collects and processes news from NewsAPI, Reddit, RSS feeds, and Korean news sources
"""

import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import hashlib

import aiohttp
try:
    import feedparser
except ImportError:
    feedparser = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from utils.logger import get_logger
from utils.exceptions import DataCollectionError

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    symbols: List[str] = None
    language: str = 'ko'
    article_id: str = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.article_id is None:
            # Generate unique ID from URL and title
            content_hash = hashlib.md5(f"{self.url}_{self.title}".encode()).hexdigest()
            self.article_id = content_hash


class TextPreprocessor:
    """Text preprocessing and cleaning utilities"""
    
    def __init__(self):
        # Korean stock code patterns
        self.korean_stock_pattern = re.compile(r'\b\d{6}\b')  # 6-digit codes
        
        # US stock symbol patterns
        self.us_stock_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Crypto patterns
        self.crypto_pattern = re.compile(r'\b(BTC|ETH|XRP|ADA|SOL|DOGE|비트코인|이더리움|리플)\b', re.IGNORECASE)
        
        # Financial keywords
        self.financial_keywords = {
            'ko': ['주식', '증시', '코스피', '코스닥', '상승', '하락', '투자', '거래량', '시가총액', '배당'],
            'en': ['stock', 'market', 'trading', 'investment', 'price', 'volume', 'dividend', 'earnings']
        }
        
        # Noise patterns to remove
        self.noise_patterns = [
            re.compile(r'<[^>]+>'),  # HTML tags
            re.compile(r'\[[^\]]+\]'),  # Brackets content
            re.compile(r'\([^)]*\)'),  # Parentheses content (some)
            re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),  # URLs
            re.compile(r'\s+'),  # Multiple whitespace
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML if BeautifulSoup is available
        if BeautifulSoup:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Apply noise pattern removal
        for pattern in self.noise_patterns:
            text = pattern.sub(' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols and codes from text"""
        symbols = []
        
        # Korean stock codes
        korean_matches = self.korean_stock_pattern.findall(text)
        symbols.extend(korean_matches)
        
        # US stock symbols (filter common words)
        us_matches = self.us_stock_pattern.findall(text)
        filtered_us = [s for s in us_matches if len(s) <= 5 and s not in ['THE', 'AND', 'FOR', 'ARE', 'BUT']]
        symbols.extend(filtered_us)
        
        # Crypto symbols
        crypto_matches = self.crypto_pattern.findall(text)
        symbols.extend(crypto_matches)
        
        return list(set(symbols))
    
    def calculate_relevance_score(self, text: str, language: str = 'ko') -> float:
        """Calculate financial relevance score"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        keywords = self.financial_keywords.get(language, [])
        
        if not keywords:
            return 0.0
        
        # Count keyword occurrences
        keyword_count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
        
        # Normalize by text length
        words = text_lower.split()
        if not words:
            return 0.0
        
        relevance_score = (keyword_count / len(words)) * 100
        return min(relevance_score, 10.0)  # Cap at 10.0


class NewsAPICollector:
    """NewsAPI.org collector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            headers = {
                'X-API-Key': self.api_key,
                'User-Agent': 'FinancialDataCollector/1.0'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def collect_financial_news(self, query: str = "finance OR stock OR crypto", 
                                   language: str = 'en', hours_back: int = 24) -> List[NewsArticle]:
        """Collect financial news from NewsAPI"""
        try:
            session = await self.get_session()
            
            # Calculate time range
            from_time = datetime.now() - timedelta(hours=hours_back)
            from_str = from_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            params = {
                'q': query,
                'language': language,
                'sortBy': 'publishedAt',
                'from': from_str,
                'pageSize': 100
            }
            
            url = f"{self.base_url}/everything"
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []
                
                data = await response.json()
                articles = []
                
                for article_data in data.get('articles', []):
                    try:
                        article = NewsArticle(
                            title=article_data.get('title', ''),
                            content=article_data.get('content', '') or article_data.get('description', ''),
                            url=article_data.get('url', ''),
                            source=article_data.get('source', {}).get('name', 'NewsAPI'),
                            published_at=datetime.fromisoformat(
                                article_data.get('publishedAt', '').replace('Z', '+00:00')
                            ),
                            author=article_data.get('author'),
                            language=language
                        )
                        articles.append(article)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse NewsAPI article: {e}")
                
                logger.info(f"Collected {len(articles)} articles from NewsAPI")
                return articles
        
        except Exception as e:
            logger.error(f"NewsAPI collection error: {e}")
            return []
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


class RedditCollector:
    """Reddit financial subreddits collector"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.session = None
        self.access_token = None
        self.financial_subreddits = [
            'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
            'financialindependence', 'StockMarket', 'SecurityAnalysis',
            'Korea_Stock', 'CryptoCurrency', 'Bitcoin'
        ]
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            headers = {'User-Agent': self.user_agent}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def authenticate(self) -> bool:
        """Authenticate with Reddit API"""
        try:
            session = await self.get_session()
            
            auth_data = {
                'grant_type': 'client_credentials'
            }
            
            auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
            
            async with session.post('https://www.reddit.com/api/v1/access_token',
                                  data=auth_data, auth=auth) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get('access_token')
                    return True
                else:
                    logger.error(f"Reddit auth failed: {response.status}")
                    return False
        
        except Exception as e:
            logger.error(f"Reddit authentication error: {e}")
            return False
    
    async def collect_subreddit_posts(self, subreddit: str, limit: int = 50) -> List[NewsArticle]:
        """Collect posts from a financial subreddit"""
        if not self.access_token:
            if not await self.authenticate():
                return []
        
        try:
            session = await self.get_session()
            headers = {'Authorization': f'bearer {self.access_token}'}
            
            url = f"https://oauth.reddit.com/r/{subreddit}/hot"
            params = {'limit': limit}
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Reddit API error: {response.status}")
                    return []
                
                data = await response.json()
                articles = []
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    try:
                        # Skip removed or deleted posts
                        if post_data.get('removed_by_category') or post_data.get('selftext') == '[deleted]':
                            continue
                        
                        article = NewsArticle(
                            title=post_data.get('title', ''),
                            content=post_data.get('selftext', '') or post_data.get('url', ''),
                            url=f"https://reddit.com{post_data.get('permalink', '')}",
                            source=f"Reddit r/{subreddit}",
                            published_at=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                            author=post_data.get('author'),
                            category='social',
                            language='en'
                        )
                        articles.append(article)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse Reddit post: {e}")
                
                logger.info(f"Collected {len(articles)} posts from r/{subreddit}")
                return articles
        
        except Exception as e:
            logger.error(f"Reddit collection error for r/{subreddit}: {e}")
            return []
    
    async def collect_all_subreddits(self) -> List[NewsArticle]:
        """Collect from all financial subreddits"""
        all_articles = []
        
        for subreddit in self.financial_subreddits:
            articles = await self.collect_subreddit_posts(subreddit)
            all_articles.extend(articles)
            await asyncio.sleep(1)  # Rate limiting
        
        return all_articles
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


class RSSCollector:
    """RSS feed collector for financial news"""
    
    def __init__(self):
        self.financial_feeds = {
            'reuters_finance': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
        }
    
    async def collect_rss_feed(self, feed_url: str, source_name: str) -> List[NewsArticle]:
        """Collect articles from RSS feed"""
        if not feedparser:
            logger.warning("feedparser not available, skipping RSS collection")
            return []
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries:
                try:
                    # Parse published date
                    published_at = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_at = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_at = datetime(*entry.updated_parsed[:6])
                    
                    article = NewsArticle(
                        title=entry.get('title', ''),
                        content=entry.get('summary', '') or entry.get('description', ''),
                        url=entry.get('link', ''),
                        source=source_name,
                        published_at=published_at,
                        author=entry.get('author'),
                        language='en'
                    )
                    articles.append(article)
                
                except Exception as e:
                    logger.warning(f"Failed to parse RSS entry: {e}")
            
            logger.info(f"Collected {len(articles)} articles from {source_name}")
            return articles
        
        except Exception as e:
            logger.error(f"RSS collection error for {source_name}: {e}")
            return []
    
    async def collect_all_feeds(self) -> List[NewsArticle]:
        """Collect from all RSS feeds"""
        all_articles = []
        
        tasks = []
        for source_name, feed_url in self.financial_feeds.items():
            task = self.collect_rss_feed(feed_url, source_name)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        return all_articles


class KoreanNewsCollector:
    """Korean financial news collector (Naver Finance, etc.)"""
    
    def __init__(self):
        self.session = None
        self.naver_finance_urls = {
            'market_news': 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258',
            'stock_news': 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=259',
            'economy_news': 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=260'
        }
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def collect_naver_finance_news(self, category: str = 'market_news') -> List[NewsArticle]:
        """Collect news from Naver Finance"""
        if not BeautifulSoup:
            logger.warning("BeautifulSoup not available, skipping Korean news collection")
            return []
        
        try:
            session = await self.get_session()
            url = self.naver_finance_urls.get(category, self.naver_finance_urls['market_news'])
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Naver Finance error: {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                articles = []
                news_items = soup.find_all('tr', class_='')  # Naver finance news rows
                
                for item in news_items:
                    try:
                        title_link = item.find('a')
                        if not title_link:
                            continue
                        
                        title = title_link.get_text(strip=True)
                        relative_url = title_link.get('href', '')
                        
                        if relative_url:
                            article_url = urljoin('https://finance.naver.com', relative_url)
                        else:
                            continue
                        
                        # Get article date
                        date_elem = item.find('td', class_='date')
                        published_at = datetime.now()
                        
                        if date_elem:
                            date_text = date_elem.get_text(strip=True)
                            try:
                                # Parse Korean date format
                                if '.' in date_text:
                                    # Format: 2023.12.29
                                    published_at = datetime.strptime(date_text, '%Y.%m.%d')
                            except:
                                pass
                        
                        # Get source
                        source_elem = item.find('td', class_='info')
                        source = source_elem.get_text(strip=True) if source_elem else 'Naver Finance'
                        
                        article = NewsArticle(
                            title=title,
                            content="",  # Would need to fetch full article
                            url=article_url,
                            source=source,
                            published_at=published_at,
                            category=category,
                            language='ko'
                        )
                        articles.append(article)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse Naver news item: {e}")
                
                logger.info(f"Collected {len(articles)} articles from Naver Finance {category}")
                return articles
        
        except Exception as e:
            logger.error(f"Naver Finance collection error: {e}")
            return []
    
    async def collect_all_categories(self) -> List[NewsArticle]:
        """Collect from all Naver Finance categories"""
        all_articles = []
        
        for category in self.naver_finance_urls.keys():
            articles = await self.collect_naver_finance_news(category)
            all_articles.extend(articles)
            await asyncio.sleep(2)  # Respectful rate limiting
        
        return all_articles
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


class NewsCollector:
    """Main news collector orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preprocessor = TextPreprocessor()
        self.collected_articles = []
        
        # Initialize collectors
        self.collectors = {}
        
        # NewsAPI
        if self.config.get('newsapi_key'):
            self.collectors['newsapi'] = NewsAPICollector(self.config['newsapi_key'])
        
        # Reddit
        if all(k in self.config for k in ['reddit_client_id', 'reddit_client_secret', 'reddit_user_agent']):
            self.collectors['reddit'] = RedditCollector(
                self.config['reddit_client_id'],
                self.config['reddit_client_secret'],
                self.config['reddit_user_agent']
            )
        
        # RSS (no config needed)
        self.collectors['rss'] = RSSCollector()
        
        # Korean news (no config needed)
        self.collectors['korean'] = KoreanNewsCollector()
    
    async def collect_all_news(self, hours_back: int = 24) -> List[NewsArticle]:
        """Collect news from all configured sources"""
        all_articles = []
        
        # Collect from all sources
        collection_tasks = []
        
        # NewsAPI
        if 'newsapi' in self.collectors:
            collection_tasks.append(
                self.collectors['newsapi'].collect_financial_news(hours_back=hours_back)
            )
        
        # Reddit
        if 'reddit' in self.collectors:
            collection_tasks.append(
                self.collectors['reddit'].collect_all_subreddits()
            )
        
        # RSS
        collection_tasks.append(
            self.collectors['rss'].collect_all_feeds()
        )
        
        # Korean news
        collection_tasks.append(
            self.collectors['korean'].collect_all_categories()
        )
        
        # Execute all collections
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Collection error: {result}")
        
        # Process articles
        processed_articles = []
        seen_urls = set()
        
        for article in all_articles:
            # Deduplicate by URL
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)
            
            # Clean and process text
            article.title = self.preprocessor.clean_text(article.title)
            article.content = self.preprocessor.clean_text(article.content)
            
            # Extract symbols
            full_text = f"{article.title} {article.content}"
            article.symbols = self.preprocessor.extract_symbols(full_text)
            
            # Calculate relevance score
            article.relevance_score = self.preprocessor.calculate_relevance_score(
                full_text, article.language
            )
            
            # Only keep relevant articles
            if article.relevance_score > 0.5:  # Threshold for relevance
                processed_articles.append(article)
        
        self.collected_articles = processed_articles
        logger.info(f"Collected and processed {len(processed_articles)} relevant articles")
        
        return processed_articles
    
    async def get_symbol_news(self, symbol: str, hours_back: int = 24) -> List[NewsArticle]:
        """Get news articles related to a specific symbol"""
        if not self.collected_articles:
            await self.collect_all_news(hours_back)
        
        relevant_articles = []
        symbol_upper = symbol.upper()
        
        for article in self.collected_articles:
            # Check if symbol is mentioned
            if (symbol_upper in article.symbols or
                symbol.lower() in article.title.lower() or
                symbol.lower() in article.content.lower()):
                relevant_articles.append(article)
        
        return relevant_articles
    
    async def close(self):
        """Close all collector sessions"""
        for collector in self.collectors.values():
            if hasattr(collector, 'close'):
                await collector.close()


# Global collector instance
news_collector = None


async def initialize_news_collector(config: Dict[str, Any]) -> NewsCollector:
    """Initialize global news collector with configuration"""
    global news_collector
    news_collector = NewsCollector(config)
    return news_collector


# Usage example
async def example_usage():
    """Example usage of news collector"""
    config = {
        'newsapi_key': 'your_newsapi_key',
        'reddit_client_id': 'your_reddit_client_id',
        'reddit_client_secret': 'your_reddit_client_secret',
        'reddit_user_agent': 'FinancialDataCollector/1.0'
    }
    
    collector = await initialize_news_collector(config)
    
    # Collect all news
    articles = await collector.collect_all_news(hours_back=24)
    print(f"Collected {len(articles)} articles")
    
    # Get news for specific symbol
    btc_news = await collector.get_symbol_news('BTC', hours_back=48)
    print(f"Found {len(btc_news)} BTC-related articles")
    
    await collector.close()


if __name__ == "__main__":
    asyncio.run(example_usage())