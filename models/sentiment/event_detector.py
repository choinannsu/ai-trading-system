"""
Financial Event Detection and Impact Prediction
Automatic detection of market-moving events and impact assessment
"""

import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
from collections import defaultdict, Counter
from enum import Enum
import asyncio

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class EventType(Enum):
    """Types of financial events"""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    MONETARY_POLICY = "monetary_policy"
    GEOPOLITICAL = "geopolitical"
    EARNINGS_GUIDANCE = "earnings_guidance"
    ANALYST_RATING = "analyst_rating"
    PRODUCT_LAUNCH = "product_launch"
    LEGAL_ISSUE = "legal_issue"
    LEADERSHIP_CHANGE = "leadership_change"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    BANKRUPTCY = "bankruptcy"
    IPO = "ipo"
    ECONOMIC_DATA = "economic_data"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_SECURITY = "cyber_security"
    OTHER = "other"


class EventSeverity(Enum):
    """Severity levels for events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EventConfig:
    """Configuration for event detection"""
    # Detection settings
    min_news_threshold: int = 3
    time_window_hours: int = 6
    similarity_threshold: float = 0.3
    impact_threshold: float = 0.1
    
    # Text analysis
    max_features: int = 1000
    min_df: int = 2
    max_df: float = 0.8
    
    # Historical matching
    lookback_days: int = 365
    min_historical_events: int = 5
    similarity_weight: float = 0.7
    
    # Impact prediction
    volatility_threshold: float = 0.02
    volume_threshold: float = 1.5
    price_impact_threshold: float = 0.05
    
    # Event filtering
    min_confidence: float = 0.6
    max_events_per_day: int = 20


@dataclass 
class EventSignal:
    """Signal indicating potential event"""
    timestamp: datetime
    event_type: EventType
    description: str
    confidence: float
    sources: List[str]
    affected_entities: List[str]
    keywords: List[str]
    news_count: int
    sentiment_shift: float


@dataclass
class HistoricalEvent:
    """Historical event for pattern matching"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    description: str
    affected_entities: List[str]
    market_impact: Dict[str, float]  # price changes, volatility, volume
    severity: EventSeverity
    duration_hours: int
    keywords: List[str]


@dataclass
class EventImpactPrediction:
    """Predicted impact of an event"""
    event_signal: EventSignal
    predicted_price_impact: float
    predicted_volatility: float
    predicted_volume_ratio: float
    confidence: float
    similar_historical_events: List[HistoricalEvent]
    risk_level: str
    recommended_actions: List[str]
    time_horizon_hours: int


@dataclass
class EventResult:
    """Complete event detection result"""
    detected_events: List[EventSignal]
    impact_predictions: List[EventImpactPrediction]
    market_stress_level: float
    event_clusters: Dict[str, List[EventSignal]]
    trending_themes: List[str]
    analysis_timestamp: datetime


class EventPatternMatcher:
    """Match current events with historical patterns"""
    
    def __init__(self):
        # Event pattern keywords
        self.event_patterns = {
            EventType.EARNINGS: {
                'keywords': [
                    'earnings', 'quarterly results', 'eps', 'revenue', 'profit',
                    'guidance', 'outlook', 'beat estimates', 'miss estimates'
                ],
                'regex_patterns': [
                    r'Q[1-4]\s+\d{4}\s+earnings',
                    r'reports?\s+\$[\d.,]+[BM]?\s+revenue',
                    r'EPS\s+of\s+\$[\d.]+',
                    r'beat\s+by\s+\$[\d.]+'
                ]
            },
            EventType.MERGER_ACQUISITION: {
                'keywords': [
                    'merger', 'acquisition', 'buyout', 'takeover', 'deal',
                    'acquire', 'purchase', 'combine', 'consolidation'
                ],
                'regex_patterns': [
                    r'acquire[sd]?\s+.+\s+for\s+\$[\d.,]+[BM]',
                    r'merger\s+worth\s+\$[\d.,]+[BM]',
                    r'takeover\s+bid',
                    r'buyout\s+offer'
                ]
            },
            EventType.MONETARY_POLICY: {
                'keywords': [
                    'federal reserve', 'fed', 'interest rate', 'monetary policy',
                    'fomc', 'powell', 'rate cut', 'rate hike', 'tapering'
                ],
                'regex_patterns': [
                    r'fed\s+raises?\s+rates?',
                    r'fed\s+cuts?\s+rates?',
                    r'interest\s+rate.+basis\s+points',
                    r'fomc\s+meeting'
                ]
            },
            EventType.REGULATORY: {
                'keywords': [
                    'sec', 'fda', 'regulation', 'investigation', 'fine',
                    'lawsuit', 'settlement', 'compliance', 'violation'
                ],
                'regex_patterns': [
                    r'sec\s+investigate[sd]?',
                    r'fined?\s+\$[\d.,]+[BM]',
                    r'settlement\s+of\s+\$[\d.,]+[BM]',
                    r'regulatory\s+approval'
                ]
            },
            EventType.ANALYST_RATING: {
                'keywords': [
                    'upgrade', 'downgrade', 'rating', 'price target',
                    'outperform', 'underperform', 'buy', 'sell', 'hold'
                ],
                'regex_patterns': [
                    r'upgrade[sd]?\s+to\s+\w+',
                    r'downgrade[sd]?\s+to\s+\w+',
                    r'price\s+target\s+\$[\d.]+',
                    r'analyst\s+rating'
                ]
            }
        }
    
    def detect_event_type(self, text: str) -> List[Tuple[EventType, float]]:
        """Detect event types in text"""
        text_lower = text.lower()
        detected_types = []
        
        for event_type, patterns in self.event_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_count = sum(1 for keyword in patterns['keywords'] 
                              if keyword in text_lower)
            score += keyword_count * 0.3
            
            # Check regex patterns
            regex_count = sum(1 for pattern in patterns['regex_patterns']
                            if re.search(pattern, text_lower))
            score += regex_count * 0.7
            
            if score > 0:
                confidence = min(1.0, score)
                detected_types.append((event_type, confidence))
        
        # Sort by confidence
        detected_types.sort(key=lambda x: x[1], reverse=True)
        return detected_types
    
    def extract_entities(self, text: str, event_type: EventType) -> List[str]:
        """Extract affected entities based on event type"""
        entities = []
        
        # Extract stock tickers
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        tickers = re.findall(ticker_pattern, text)
        entities.extend(tickers)
        
        # Extract company names
        company_pattern = r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:Inc|Corp|Ltd|Co|Company|Group)\b'
        companies = re.findall(company_pattern, text)
        entities.extend(companies)
        
        # Event-specific entity extraction
        if event_type == EventType.MONETARY_POLICY:
            fed_entities = ['Federal Reserve', 'Fed', 'FOMC', 'Jerome Powell']
            entities.extend([entity for entity in fed_entities if entity.lower() in text.lower()])
        
        return list(set(entities))


class EventClusterer:
    """Cluster similar events together"""
    
    def __init__(self, config: EventConfig):
        self.config = config
        self.vectorizer = None
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                stop_words='english'
            )
    
    def cluster_events(self, events: List[EventSignal]) -> Dict[str, List[EventSignal]]:
        """Cluster similar events together"""
        if not SKLEARN_AVAILABLE or len(events) < 2:
            return {'cluster_0': events}
        
        try:
            # Prepare text data
            texts = [f"{event.description} {' '.join(event.keywords)}" for event in events]
            
            # Vectorize text
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Perform clustering
            clustering = DBSCAN(
                eps=1-self.config.similarity_threshold,
                min_samples=2,
                metric='cosine'
            )
            
            labels = clustering.fit_predict(tfidf_matrix.toarray())
            
            # Group events by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                cluster_name = f"cluster_{label}" if label != -1 else "outliers"
                clusters[cluster_name].append(events[i])
            
            return dict(clusters)
            
        except Exception as e:
            logger.warning(f"Event clustering failed: {e}")
            return {'cluster_0': events}


class HistoricalEventDatabase:
    """Simplified historical event database for pattern matching"""
    
    def __init__(self):
        self.events = []
        self._initialize_sample_events()
    
    def _initialize_sample_events(self):
        """Initialize with sample historical events"""
        # Sample events for demonstration
        sample_events = [
            {
                'event_id': 'earnings_001',
                'timestamp': datetime.now() - timedelta(days=90),
                'event_type': EventType.EARNINGS,
                'description': 'Apple reports strong Q3 earnings',
                'affected_entities': ['AAPL'],
                'market_impact': {'price_change': 0.05, 'volatility': 0.03, 'volume_ratio': 2.1},
                'severity': EventSeverity.MEDIUM,
                'duration_hours': 24,
                'keywords': ['earnings', 'revenue', 'iphone', 'services']
            },
            {
                'event_id': 'fed_001', 
                'timestamp': datetime.now() - timedelta(days=60),
                'event_type': EventType.MONETARY_POLICY,
                'description': 'Fed raises interest rates by 0.25%',
                'affected_entities': ['SPY', 'QQQ', 'Market'],
                'market_impact': {'price_change': -0.02, 'volatility': 0.04, 'volume_ratio': 1.8},
                'severity': EventSeverity.HIGH,
                'duration_hours': 48,
                'keywords': ['fed', 'interest rate', 'monetary policy', 'inflation']
            },
            {
                'event_id': 'merger_001',
                'timestamp': datetime.now() - timedelta(days=120),
                'event_type': EventType.MERGER_ACQUISITION,
                'description': 'Microsoft acquires gaming company',
                'affected_entities': ['MSFT'],
                'market_impact': {'price_change': 0.08, 'volatility': 0.025, 'volume_ratio': 3.2},
                'severity': EventSeverity.HIGH,
                'duration_hours': 72,
                'keywords': ['acquisition', 'gaming', 'deal', 'microsoft']
            }
        ]
        
        for event_data in sample_events:
            event = HistoricalEvent(**event_data)
            self.events.append(event)
    
    def add_event(self, event: HistoricalEvent):
        """Add new historical event"""
        self.events.append(event)
    
    def find_similar_events(self, current_event: EventSignal, 
                          lookback_days: int = 365, 
                          min_similarity: float = 0.3) -> List[HistoricalEvent]:
        """Find historically similar events"""
        similar_events = []
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        for historical_event in self.events:
            if historical_event.timestamp < cutoff_date:
                continue
            
            # Calculate similarity
            similarity = self._calculate_event_similarity(current_event, historical_event)
            
            if similarity >= min_similarity:
                similar_events.append((historical_event, similarity))
        
        # Sort by similarity
        similar_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in similar_events]
    
    def _calculate_event_similarity(self, current: EventSignal, 
                                  historical: HistoricalEvent) -> float:
        """Calculate similarity between current and historical event"""
        similarity = 0.0
        
        # Event type similarity
        if current.event_type == historical.event_type:
            similarity += 0.4
        
        # Entity overlap
        current_entities = set([entity.upper() for entity in current.affected_entities])
        historical_entities = set([entity.upper() for entity in historical.affected_entities])
        
        if current_entities and historical_entities:
            entity_similarity = len(current_entities & historical_entities) / len(current_entities | historical_entities)
            similarity += 0.3 * entity_similarity
        
        # Keyword overlap
        current_keywords = set([kw.lower() for kw in current.keywords])
        historical_keywords = set([kw.lower() for kw in historical.keywords])
        
        if current_keywords and historical_keywords:
            keyword_similarity = len(current_keywords & historical_keywords) / len(current_keywords | historical_keywords)
            similarity += 0.3 * keyword_similarity
        
        return similarity


class EventDetector:
    """Main event detection and impact prediction system"""
    
    def __init__(self, config: EventConfig = None):
        self.config = config or EventConfig()
        self.pattern_matcher = EventPatternMatcher()
        self.clusterer = EventClusterer(self.config)
        self.historical_db = HistoricalEventDatabase()
        
        # Cache for recent analysis
        self.recent_events = []
    
    def detect_events_from_news(self, news_data: List[Dict[str, Any]]) -> List[EventSignal]:
        """Detect events from news data"""
        detected_events = []
        
        # Group news by time windows
        time_windows = self._group_news_by_time(news_data)
        
        for window_start, window_news in time_windows.items():
            if len(window_news) < self.config.min_news_threshold:
                continue
            
            # Analyze news in this time window
            window_events = self._analyze_news_window(window_news, window_start)
            detected_events.extend(window_events)
        
        # Filter by confidence
        filtered_events = [event for event in detected_events 
                         if event.confidence >= self.config.min_confidence]
        
        # Limit number of events
        filtered_events.sort(key=lambda x: x.confidence, reverse=True)
        return filtered_events[:self.config.max_events_per_day]
    
    def _group_news_by_time(self, news_data: List[Dict[str, Any]]) -> Dict[datetime, List[Dict]]:
        """Group news by time windows"""
        time_windows = defaultdict(list)
        
        for news in news_data:
            # Extract timestamp
            if 'timestamp' in news:
                timestamp = news['timestamp']
            elif 'published_at' in news:
                timestamp = news['published_at']
            else:
                timestamp = datetime.now()
            
            # Round to time window
            window_start = timestamp.replace(
                minute=0, second=0, microsecond=0
            )
            # Group by N-hour windows
            window_hours = window_start.hour // self.config.time_window_hours * self.config.time_window_hours
            window_start = window_start.replace(hour=window_hours)
            
            time_windows[window_start].append(news)
        
        return time_windows
    
    def _analyze_news_window(self, news_items: List[Dict[str, Any]], 
                           window_start: datetime) -> List[EventSignal]:
        """Analyze news items in a time window for events"""
        events = []
        
        # Combine all text
        all_text = ""
        sources = []
        
        for news in news_items:
            title = news.get('title', '')
            content = news.get('content', '')
            source = news.get('source', 'unknown')
            
            all_text += f"{title}. {content} "
            sources.append(source)
        
        # Detect event types
        detected_types = self.pattern_matcher.detect_event_type(all_text)
        
        for event_type, confidence in detected_types:
            if confidence < self.config.min_confidence:
                continue
            
            # Extract entities and keywords
            entities = self.pattern_matcher.extract_entities(all_text, event_type)
            keywords = self._extract_keywords(all_text, event_type)
            
            # Calculate sentiment shift (simplified)
            sentiment_shift = self._calculate_sentiment_shift(news_items)
            
            # Create event signal
            event = EventSignal(
                timestamp=window_start,
                event_type=event_type,
                description=self._generate_event_description(event_type, entities, all_text),
                confidence=confidence,
                sources=list(set(sources)),
                affected_entities=entities,
                keywords=keywords,
                news_count=len(news_items),
                sentiment_shift=sentiment_shift
            )
            
            events.append(event)
        
        return events
    
    def _extract_keywords(self, text: str, event_type: EventType) -> List[str]:
        """Extract relevant keywords from text"""
        # Get event-specific keywords
        event_keywords = self.pattern_matcher.event_patterns.get(event_type, {}).get('keywords', [])
        
        text_lower = text.lower()
        found_keywords = [kw for kw in event_keywords if kw in text_lower]
        
        # Add additional keywords using simple frequency analysis
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        word_counts = Counter(words)
        
        # Get common words (excluding very common ones)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        additional_keywords = [word for word, count in word_counts.most_common(10) 
                             if word not in common_words and len(word) > 3]
        
        found_keywords.extend(additional_keywords[:5])
        return found_keywords
    
    def _calculate_sentiment_shift(self, news_items: List[Dict[str, Any]]) -> float:
        """Calculate sentiment shift from news items"""
        # Simplified sentiment calculation
        positive_words = ['up', 'gain', 'rise', 'growth', 'profit', 'strong', 'beat']
        negative_words = ['down', 'loss', 'fall', 'decline', 'weak', 'miss', 'crash']
        
        total_sentiment = 0
        for news in news_items:
            text = f"{news.get('title', '')} {news.get('content', '')}".lower()
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                total_sentiment += sentiment
        
        return total_sentiment / max(len(news_items), 1)
    
    def _generate_event_description(self, event_type: EventType, 
                                  entities: List[str], text: str) -> str:
        """Generate human-readable event description"""
        entity_str = ", ".join(entities[:3]) if entities else "Market"
        
        if event_type == EventType.EARNINGS:
            return f"{entity_str} earnings announcement"
        elif event_type == EventType.MERGER_ACQUISITION:
            return f"{entity_str} merger/acquisition activity"
        elif event_type == EventType.MONETARY_POLICY:
            return f"Federal Reserve monetary policy update"
        elif event_type == EventType.REGULATORY:
            return f"{entity_str} regulatory development"
        elif event_type == EventType.ANALYST_RATING:
            return f"{entity_str} analyst rating change"
        else:
            return f"{entity_str} {event_type.value} event"
    
    def predict_event_impact(self, event: EventSignal) -> EventImpactPrediction:
        """Predict market impact of detected event"""
        # Find similar historical events
        similar_events = self.historical_db.find_similar_events(
            event, self.config.lookback_days, 0.3
        )
        
        # Calculate predicted impact based on historical data
        if similar_events:
            avg_price_impact = np.mean([e.market_impact['price_change'] for e in similar_events])
            avg_volatility = np.mean([e.market_impact['volatility'] for e in similar_events])
            avg_volume_ratio = np.mean([e.market_impact['volume_ratio'] for e in similar_events])
            
            # Adjust based on current event confidence and severity
            confidence_multiplier = event.confidence
            sentiment_multiplier = 1.0 + abs(event.sentiment_shift)
            
            predicted_price_impact = avg_price_impact * confidence_multiplier * sentiment_multiplier
            predicted_volatility = avg_volatility * confidence_multiplier
            predicted_volume_ratio = avg_volume_ratio * confidence_multiplier
            
            # Calculate overall confidence
            confidence = min(0.9, len(similar_events) / 5 * event.confidence)
        else:
            # Default predictions when no historical data
            base_impact = 0.01 * event.confidence
            predicted_price_impact = base_impact * (1 + event.sentiment_shift)
            predicted_volatility = base_impact * 2
            predicted_volume_ratio = 1.0 + base_impact * 10
            confidence = 0.3
        
        # Determine risk level
        if abs(predicted_price_impact) > 0.05 or predicted_volatility > 0.04:
            risk_level = "High"
        elif abs(predicted_price_impact) > 0.02 or predicted_volatility > 0.02:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(event, predicted_price_impact, risk_level)
        
        # Estimate time horizon
        time_horizon = self._estimate_time_horizon(event.event_type, similar_events)
        
        return EventImpactPrediction(
            event_signal=event,
            predicted_price_impact=predicted_price_impact,
            predicted_volatility=predicted_volatility,
            predicted_volume_ratio=predicted_volume_ratio,
            confidence=confidence,
            similar_historical_events=similar_events,
            risk_level=risk_level,
            recommended_actions=recommendations,
            time_horizon_hours=time_horizon
        )
    
    def _generate_recommendations(self, event: EventSignal, 
                                predicted_impact: float, risk_level: str) -> List[str]:
        """Generate trading recommendations based on event"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.append("Consider reducing position sizes")
            recommendations.append("Set tighter stop losses")
            recommendations.append("Monitor volatility closely")
        
        if abs(predicted_impact) > 0.03:
            if predicted_impact > 0:
                recommendations.append("Potential buying opportunity")
            else:
                recommendations.append("Consider hedging long positions")
        
        if event.event_type == EventType.EARNINGS:
            recommendations.append("Watch for earnings surprises")
            recommendations.append("Monitor guidance updates")
        
        elif event.event_type == EventType.MONETARY_POLICY:
            recommendations.append("Review interest rate sensitive positions")
            recommendations.append("Consider bond market implications")
        
        return recommendations
    
    def _estimate_time_horizon(self, event_type: EventType, 
                             similar_events: List[HistoricalEvent]) -> int:
        """Estimate how long event impact will last"""
        if similar_events:
            avg_duration = np.mean([e.duration_hours for e in similar_events])
            return int(avg_duration)
        
        # Default durations by event type
        default_durations = {
            EventType.EARNINGS: 24,
            EventType.MONETARY_POLICY: 48,
            EventType.MERGER_ACQUISITION: 72,
            EventType.REGULATORY: 48,
            EventType.ANALYST_RATING: 12,
            EventType.ECONOMIC_DATA: 24
        }
        
        return default_durations.get(event_type, 24)
    
    def analyze_market_stress(self, events: List[EventSignal]) -> float:
        """Calculate overall market stress level from events"""
        if not events:
            return 0.0
        
        stress_factors = []
        
        for event in events:
            # Base stress from event type
            event_stress = {
                EventType.MONETARY_POLICY: 0.8,
                EventType.GEOPOLITICAL: 0.7,
                EventType.REGULATORY: 0.6,
                EventType.EARNINGS: 0.3,
                EventType.MERGER_ACQUISITION: 0.4
            }.get(event.event_type, 0.2)
            
            # Adjust by confidence and sentiment
            adjusted_stress = event_stress * event.confidence * (1 + abs(event.sentiment_shift))
            stress_factors.append(adjusted_stress)
        
        # Calculate weighted average stress
        return min(1.0, np.mean(stress_factors) * len(events) / 10)
    
    def detect_and_analyze(self, news_data: List[Dict[str, Any]]) -> EventResult:
        """Complete event detection and analysis pipeline"""
        # Detect events
        detected_events = self.detect_events_from_news(news_data)
        
        # Predict impacts
        impact_predictions = []
        for event in detected_events:
            prediction = self.predict_event_impact(event)
            impact_predictions.append(prediction)
        
        # Cluster events
        event_clusters = self.clusterer.cluster_events(detected_events)
        
        # Calculate market stress
        market_stress = self.analyze_market_stress(detected_events)
        
        # Extract trending themes
        all_keywords = []
        for event in detected_events:
            all_keywords.extend(event.keywords)
        
        trending_themes = [theme for theme, count in Counter(all_keywords).most_common(10)]
        
        return EventResult(
            detected_events=detected_events,
            impact_predictions=impact_predictions,
            market_stress_level=market_stress,
            event_clusters=event_clusters,
            trending_themes=trending_themes,
            analysis_timestamp=datetime.now()
        )


# Generate sample news data for testing
def generate_sample_news_events(count: int = 15) -> List[Dict[str, Any]]:
    """Generate sample news data with potential events"""
    sample_news = [
        {
            'title': 'Apple Reports Record Q4 Earnings',
            'content': 'Apple Inc. reported quarterly earnings that beat analyst estimates with revenue of $89.5 billion...',
            'timestamp': datetime.now() - timedelta(hours=2),
            'source': 'reuters'
        },
        {
            'title': 'Federal Reserve Signals Rate Cut in December',
            'content': 'Federal Reserve Chairman Jerome Powell indicated the central bank may cut interest rates...',
            'timestamp': datetime.now() - timedelta(hours=1),
            'source': 'bloomberg'
        },
        {
            'title': 'Microsoft Acquires AI Startup for $2.1 Billion',
            'content': 'Microsoft announced the acquisition of an artificial intelligence startup...',
            'timestamp': datetime.now() - timedelta(hours=3),
            'source': 'cnbc'
        },
        {
            'title': 'SEC Investigates Tech Giants Over Privacy Practices',
            'content': 'The Securities and Exchange Commission launched an investigation into major technology companies...',
            'timestamp': datetime.now() - timedelta(hours=4),
            'source': 'marketwatch'
        },
        {
            'title': 'Goldman Sachs Upgrades Tesla to Buy',
            'content': 'Goldman Sachs analysts upgraded Tesla stock to Buy with a price target of $350...',
            'timestamp': datetime.now() - timedelta(hours=5),
            'source': 'investing.com'
        }
    ]
    
    # Repeat with variations for more data
    extended_news = []
    for i in range(count):
        news = sample_news[i % len(sample_news)].copy()
        news['timestamp'] = datetime.now() - timedelta(hours=i)
        extended_news.append(news)
    
    return extended_news


# Factory function
def create_event_detector(config: Dict[str, Any] = None) -> EventDetector:
    """Create event detector with configuration"""
    if config:
        event_config = EventConfig(**config)
    else:
        event_config = EventConfig()
    
    return EventDetector(event_config)


# Export classes and functions
__all__ = [
    'EventDetector',
    'EventConfig',
    'EventSignal',
    'EventResult',
    'EventImpactPrediction',
    'HistoricalEvent',
    'EventType',
    'EventSeverity',
    'EventPatternMatcher',
    'HistoricalEventDatabase',
    'create_event_detector',
    'generate_sample_news_events'
]