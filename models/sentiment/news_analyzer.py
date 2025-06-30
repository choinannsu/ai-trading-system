"""
Financial News Sentiment Analyzer
FinBERT-based financial news analysis with Korean language support
"""

import numpy as np
import pandas as pd
import re
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
import os
from collections import defaultdict
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import requests
import time

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

from utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class FinancialNewsConfig:
    """Configuration for financial news analysis"""
    # Model settings
    finbert_model: str = "ProsusAI/finbert"
    kobert_model: str = "kykim/bert-kor-base"
    use_korean_model: bool = True
    max_length: int = 512
    batch_size: int = 16
    
    # Analysis settings
    sentiment_threshold: float = 0.1
    impact_weight: float = 0.7
    time_decay_hours: int = 24
    
    # News sources
    sources: List[str] = None
    keywords: List[str] = None
    
    # Cache settings
    cache_duration_hours: int = 1
    max_cache_size: int = 10000
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = [
                'reuters', 'bloomberg', 'cnbc', 'marketwatch', 
                'investing.com', 'yahoo_finance', 'naver_finance', 'hankyung'
            ]
        if self.keywords is None:
            self.keywords = [
                'stock', 'market', 'economy', 'finance', 'trading',
                '주식', '증시', '경제', '금융', '투자', '시장'
            ]


@dataclass
class NewsItem:
    """Individual news item"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    language: str = 'en'
    author: str = None
    category: str = None


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    label: str  # 'positive', 'negative', 'neutral'


@dataclass
class NewsAnalysisResult:
    """Complete news analysis result"""
    news_item: NewsItem
    sentiment: SentimentScore
    impact_score: float
    key_entities: List[str]
    topics: List[str]
    financial_keywords: List[str]
    time_weight: float
    source_credibility: float
    analysis_timestamp: datetime


class FinancialKeywordExtractor:
    """Extract financial keywords and entities"""
    
    def __init__(self):
        # Financial keywords dictionary
        self.financial_keywords = {
            'en': {
                'market_terms': [
                    'bull market', 'bear market', 'volatility', 'rally', 'correction',
                    'earnings', 'revenue', 'profit', 'loss', 'dividend', 'buyback',
                    'IPO', 'merger', 'acquisition', 'bankruptcy', 'restructuring'
                ],
                'indicators': [
                    'GDP', 'inflation', 'unemployment', 'interest rate', 'fed rate',
                    'CPI', 'PPI', 'retail sales', 'housing starts', 'consumer confidence'
                ],
                'actions': [
                    'buy', 'sell', 'hold', 'upgrade', 'downgrade', 'outperform',
                    'underperform', 'target price', 'recommendation', 'rating'
                ]
            },
            'ko': {
                'market_terms': [
                    '강세장', '약세장', '변동성', '랠리', '조정',
                    '실적', '매출', '이익', '손실', '배당', '자사주매입',
                    '상장', '합병', '인수', '파산', '구조조정'
                ],
                'indicators': [
                    'GDP', '인플레이션', '실업률', '금리', '기준금리',
                    '소비자물가', '생산자물가', '소매판매', '주택착공', '소비자신뢰'
                ],
                'actions': [
                    '매수', '매도', '보유', '상향', '하향', '아웃퍼폼',
                    '언더퍼폼', '목표가', '추천', '등급'
                ]
            }
        }
    
    def extract_keywords(self, text: str, language: str = 'en') -> List[str]:
        """Extract financial keywords from text"""
        keywords = []
        text_lower = text.lower()
        
        lang_keywords = self.financial_keywords.get(language, self.financial_keywords['en'])
        
        for category, terms in lang_keywords.items():
            for term in terms:
                if term.lower() in text_lower:
                    keywords.append(term)
        
        return list(set(keywords))
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract potential financial entities (company names, tickers, etc.)"""
        entities = []
        
        # Extract potential stock tickers (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        tickers = re.findall(ticker_pattern, text)
        entities.extend(tickers)
        
        # Extract company-like names (capitalized words)
        company_pattern = r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:Inc|Corp|Ltd|Co|Company|Group)\b'
        companies = re.findall(company_pattern, text)
        entities.extend(companies)
        
        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?[BMK]?'
        amounts = re.findall(money_pattern, text)
        entities.extend(amounts)
        
        return list(set(entities))


class NewsAnalyzer:
    """Financial news sentiment analyzer with FinBERT and KoBERT"""
    
    def __init__(self, config: FinancialNewsConfig = None):
        self.config = config or FinancialNewsConfig()
        self.keyword_extractor = FinancialKeywordExtractor()
        
        # Initialize models
        self.finbert_analyzer = None
        self.kobert_analyzer = None
        self._initialize_models()
        
        # Source credibility scores
        self.source_credibility = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'cnbc': 0.85,
            'marketwatch': 0.80,
            'investing.com': 0.75,
            'yahoo_finance': 0.70,
            'naver_finance': 0.80,
            'hankyung': 0.85,
            'default': 0.50
        }
        
        # Cache for analysis results
        self.analysis_cache = {}
        
    def _initialize_models(self):
        """Initialize FinBERT and KoBERT models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Using rule-based sentiment analysis.")
            return
        
        try:
            # Initialize FinBERT for English financial news
            logger.info("Loading FinBERT model...")
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model=self.config.finbert_model,
                tokenizer=self.config.finbert_model,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize KoBERT for Korean news (if available)
            if self.config.use_korean_model:
                try:
                    logger.info("Loading KoBERT model...")
                    kobert_tokenizer = BertTokenizer.from_pretrained(self.config.kobert_model)
                    kobert_model = BertForSequenceClassification.from_pretrained(self.config.kobert_model)
                    
                    self.kobert_analyzer = pipeline(
                        "sentiment-analysis",
                        model=kobert_model,
                        tokenizer=kobert_tokenizer,
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Could not load KoBERT model: {e}")
                    self.kobert_analyzer = None
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.finbert_analyzer = None
            self.kobert_analyzer = None
    
    def _detect_language(self, text: str) -> str:
        """Detect text language (simple heuristic)"""
        # Count Korean characters
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return 'en'
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio > 0.3:
            return 'ko'
        else:
            return 'en'
    
    def _preprocess_text(self, text: str, language: str = 'en') -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to max length
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length]
        
        return text
    
    def _analyze_sentiment_transformer(self, text: str, language: str = 'en') -> SentimentScore:
        """Analyze sentiment using transformer models"""
        if language == 'ko' and self.kobert_analyzer:
            analyzer = self.kobert_analyzer
        elif self.finbert_analyzer:
            analyzer = self.finbert_analyzer
        else:
            return self._analyze_sentiment_rule_based(text, language)
        
        try:
            result = analyzer(text)
            
            # Parse FinBERT output
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Convert to standard format
            if 'positive' in label:
                positive, negative, neutral = score, 0.0, 1.0 - score
                compound = score
            elif 'negative' in label:
                positive, negative, neutral = 0.0, score, 1.0 - score
                compound = -score
            else:  # neutral
                positive, negative, neutral = 0.0, 0.0, score
                compound = 0.0
            
            return SentimentScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                compound=compound,
                confidence=score,
                label=label
            )
            
        except Exception as e:
            logger.warning(f"Transformer sentiment analysis failed: {e}")
            return self._analyze_sentiment_rule_based(text, language)
    
    def _analyze_sentiment_rule_based(self, text: str, language: str = 'en') -> SentimentScore:
        """Rule-based sentiment analysis as fallback"""
        # Simple keyword-based sentiment
        positive_words = {
            'en': ['up', 'rise', 'gain', 'profit', 'growth', 'strong', 'beat', 'exceed'],
            'ko': ['상승', '증가', '이익', '성장', '강세', '돌파', '상향', '호조']
        }
        
        negative_words = {
            'en': ['down', 'fall', 'loss', 'decline', 'weak', 'miss', 'below'],
            'ko': ['하락', '감소', '손실', '하락', '약세', '하향', '부진', '악화']
        }
        
        text_lower = text.lower()
        lang_pos = positive_words.get(language, positive_words['en'])
        lang_neg = negative_words.get(language, negative_words['en'])
        
        pos_count = sum(1 for word in lang_pos if word in text_lower)
        neg_count = sum(1 for word in lang_neg if word in text_lower)
        
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return SentimentScore(0.0, 0.0, 1.0, 0.0, 0.5, 'neutral')
        
        pos_ratio = pos_count / total_sentiment_words
        neg_ratio = neg_count / total_sentiment_words
        
        compound = pos_ratio - neg_ratio
        
        if compound > 0.1:
            label = 'positive'
        elif compound < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return SentimentScore(
            positive=pos_ratio,
            negative=neg_ratio,
            neutral=max(0, 1 - pos_ratio - neg_ratio),
            compound=compound,
            confidence=min(0.8, abs(compound) + 0.5),
            label=label
        )
    
    def _calculate_impact_score(self, news_item: NewsItem, sentiment: SentimentScore) -> float:
        """Calculate news impact score"""
        # Base impact from sentiment strength
        sentiment_impact = abs(sentiment.compound) * sentiment.confidence
        
        # Source credibility weight
        source_weight = self.source_credibility.get(news_item.source, 
                                                   self.source_credibility['default'])
        
        # Time decay (more recent = higher impact)
        hours_old = (datetime.now() - news_item.published_at).total_seconds() / 3600
        time_weight = max(0.1, 1.0 - (hours_old / self.config.time_decay_hours))
        
        # Content length factor (longer articles may be more impactful)
        content_factor = min(1.0, len(news_item.content) / 1000)
        
        # Financial keyword density
        keywords = self.keyword_extractor.extract_keywords(
            news_item.title + ' ' + news_item.content, 
            news_item.language
        )
        keyword_density = len(keywords) / max(1, len(news_item.content.split()) / 100)
        keyword_factor = min(1.0, keyword_density)
        
        # Calculate final impact score
        impact_score = (
            sentiment_impact * 
            source_weight * 
            time_weight * 
            (0.7 * content_factor + 0.3 * keyword_factor)
        )
        
        return min(1.0, impact_score)
    
    def _calculate_time_weight(self, published_at: datetime) -> float:
        """Calculate time-based weight for news relevance"""
        hours_old = (datetime.now() - published_at).total_seconds() / 3600
        return max(0.1, 1.0 - (hours_old / self.config.time_decay_hours))
    
    def _extract_topics(self, text: str, language: str = 'en') -> List[str]:
        """Extract main topics from text"""
        # Simple topic extraction based on keywords
        topics = []
        
        topic_keywords = {
            'en': {
                'earnings': ['earnings', 'profit', 'revenue', 'eps'],
                'merger': ['merger', 'acquisition', 'takeover', 'deal'],
                'ipo': ['ipo', 'public', 'listing', 'debut'],
                'regulatory': ['sec', 'regulation', 'compliance', 'investigation'],
                'monetary_policy': ['fed', 'interest rate', 'monetary policy', 'inflation'],
                'trade': ['trade', 'tariff', 'export', 'import']
            },
            'ko': {
                'earnings': ['실적', '이익', '매출', 'eps'],
                'merger': ['합병', '인수', '매각', '거래'],
                'ipo': ['상장', '공모', '데뷔'],
                'regulatory': ['금감원', '규제', '컴플라이언스', '조사'],
                'monetary_policy': ['한은', '금리', '통화정책', '인플레이션'],
                'trade': ['무역', '관세', '수출', '수입']
            }
        }
        
        text_lower = text.lower()
        lang_topics = topic_keywords.get(language, topic_keywords['en'])
        
        for topic, keywords in lang_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def analyze_news(self, news_item: NewsItem) -> NewsAnalysisResult:
        """Analyze a single news item"""
        # Check cache first
        cache_key = f"{news_item.url}_{news_item.published_at}"
        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            if (datetime.now() - cached_result.analysis_timestamp).total_seconds() < \
               self.config.cache_duration_hours * 3600:
                return cached_result
        
        # Detect language if not specified
        if not news_item.language:
            news_item.language = self._detect_language(news_item.title + ' ' + news_item.content)
        
        # Preprocess text
        full_text = f"{news_item.title}. {news_item.content}"
        processed_text = self._preprocess_text(full_text, news_item.language)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment_transformer(processed_text, news_item.language)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(news_item, sentiment)
        
        # Extract keywords and entities
        financial_keywords = self.keyword_extractor.extract_keywords(
            processed_text, news_item.language
        )
        key_entities = self.keyword_extractor.extract_entities(processed_text)
        
        # Extract topics
        topics = self._extract_topics(processed_text, news_item.language)
        
        # Calculate weights
        time_weight = self._calculate_time_weight(news_item.published_at)
        source_credibility = self.source_credibility.get(
            news_item.source, self.source_credibility['default']
        )
        
        # Create result
        result = NewsAnalysisResult(
            news_item=news_item,
            sentiment=sentiment,
            impact_score=impact_score,
            key_entities=key_entities,
            topics=topics,
            financial_keywords=financial_keywords,
            time_weight=time_weight,
            source_credibility=source_credibility,
            analysis_timestamp=datetime.now()
        )
        
        # Cache result
        if len(self.analysis_cache) < self.config.max_cache_size:
            self.analysis_cache[cache_key] = result
        
        return result
    
    def analyze_news_batch(self, news_items: List[NewsItem]) -> List[NewsAnalysisResult]:
        """Analyze multiple news items efficiently"""
        results = []
        
        for news_item in news_items:
            try:
                result = self.analyze_news(news_item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing news item {news_item.url}: {e}")
                continue
        
        return results
    
    def create_sentiment_timeseries(self, analysis_results: List[NewsAnalysisResult], 
                                  time_window: str = '1H') -> pd.DataFrame:
        """Create sentiment time series from analysis results"""
        # Convert to DataFrame
        data = []
        for result in analysis_results:
            data.append({
                'timestamp': result.news_item.published_at,
                'sentiment_compound': result.sentiment.compound,
                'sentiment_positive': result.sentiment.positive,
                'sentiment_negative': result.sentiment.negative,
                'sentiment_neutral': result.sentiment.neutral,
                'impact_score': result.impact_score,
                'confidence': result.sentiment.confidence,
                'source_credibility': result.source_credibility,
                'time_weight': result.time_weight,
                'weighted_sentiment': result.sentiment.compound * result.impact_score * result.time_weight
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Resample to specified time window
        agg_functions = {
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'impact_score': 'mean',
            'confidence': 'mean',
            'source_credibility': 'mean',
            'time_weight': 'mean',
            'weighted_sentiment': 'mean'
        }
        
        df_resampled = df.resample(time_window).agg(agg_functions)
        
        # Add additional metrics
        df_resampled['news_count'] = df.resample(time_window).size()
        df_resampled['sentiment_volatility'] = df['sentiment_compound'].resample(time_window).std()
        
        # Fill NaN values
        df_resampled = df_resampled.fillna(0)
        
        return df_resampled
    
    def get_market_sentiment_summary(self, analysis_results: List[NewsAnalysisResult]) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        if not analysis_results:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'news_count': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'top_topics': [],
                'top_entities': [],
                'average_impact': 0.0
            }
        
        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0
        sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        all_topics = []
        all_entities = []
        impact_scores = []
        
        for result in analysis_results:
            weight = result.impact_score * result.time_weight * result.source_credibility
            weighted_sentiment += result.sentiment.compound * weight
            total_weight += weight
            
            sentiment_distribution[result.sentiment.label] += 1
            all_topics.extend(result.topics)
            all_entities.extend(result.key_entities)
            impact_scores.append(result.impact_score)
        
        overall_sentiment = weighted_sentiment / max(total_weight, 1e-8)
        
        # Determine overall label
        if overall_sentiment > 0.1:
            sentiment_label = 'positive'
        elif overall_sentiment < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Get top topics and entities
        from collections import Counter
        top_topics = [topic for topic, count in Counter(all_topics).most_common(10)]
        top_entities = [entity for entity, count in Counter(all_entities).most_common(10)]
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': abs(overall_sentiment),
            'news_count': len(analysis_results),
            'sentiment_distribution': sentiment_distribution,
            'top_topics': top_topics,
            'top_entities': top_entities,
            'average_impact': np.mean(impact_scores) if impact_scores else 0.0,
            'sentiment_volatility': np.std([r.sentiment.compound for r in analysis_results])
        }


# Factory function
def create_news_analyzer(config: Dict[str, Any] = None) -> NewsAnalyzer:
    """Create news analyzer with configuration"""
    if config:
        news_config = FinancialNewsConfig(**config)
    else:
        news_config = FinancialNewsConfig()
    
    return NewsAnalyzer(news_config)


# Sample news data generator for testing
def generate_sample_news(count: int = 10) -> List[NewsItem]:
    """Generate sample news items for testing"""
    sample_titles = [
        "Stock market rallies on positive earnings reports",
        "Federal Reserve signals potential interest rate cuts",
        "Tech stocks decline amid regulatory concerns",
        "Oil prices surge following supply disruption",
        "주식시장 강세, 긍정적인 실적 발표로 상승",
        "금리 인하 신호에 증시 급등",
        "Big tech earnings beat expectations significantly",
        "Inflation data comes in lower than expected",
        "한국은행 기준금리 동결 결정",
        "글로벌 경기침체 우려에 증시 하락"
    ]
    
    sample_contents = [
        "Markets showed strong performance today with major indices posting significant gains...",
        "The Federal Reserve Chairman indicated a more dovish stance in recent comments...",
        "Technology sector faces headwinds from increased regulatory scrutiny...",
        "Crude oil prices jumped more than 5% following news of supply chain disruptions...",
        "오늘 주식시장은 강한 실적 발표에 힘입어 상승세를 보였습니다...",
        "한국은행의 금리 인하 시사로 증시가 급등했습니다...",
        "Major technology companies reported earnings that exceeded analyst expectations...",
        "Latest inflation data shows cooling price pressures across multiple sectors...",
        "한국은행이 기준금리를 현 수준에서 동결하기로 결정했습니다...",
        "글로벌 경기 둔화 우려로 주요 증시가 일제히 하락했습니다..."
    ]
    
    sources = ['reuters', 'bloomberg', 'cnbc', 'naver_finance', 'hankyung']
    
    news_items = []
    for i in range(count):
        news_items.append(NewsItem(
            title=sample_titles[i % len(sample_titles)],
            content=sample_contents[i % len(sample_contents)],
            source=sources[i % len(sources)],
            url=f"https://example.com/news/{i}",
            published_at=datetime.now() - timedelta(hours=i),
            language='ko' if i % 3 == 0 else 'en'
        ))
    
    return news_items


# Export classes and functions
__all__ = [
    'NewsAnalyzer',
    'FinancialNewsConfig', 
    'NewsItem',
    'SentimentScore',
    'NewsAnalysisResult',
    'FinancialKeywordExtractor',
    'create_news_analyzer',
    'generate_sample_news'
]