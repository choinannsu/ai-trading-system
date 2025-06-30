"""
Data Quality Checker
Comprehensive data validation, cleaning, and quality assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics

from utils.logger import get_logger

logger = get_logger(__name__)


class QualityIssueType(Enum):
    """Types of data quality issues"""
    MISSING_VALUE = "missing_value"
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    INCONSISTENT = "inconsistent"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"


@dataclass
class QualityIssue:
    """Individual data quality issue"""
    issue_type: QualityIssueType
    field: str
    value: Any
    row_index: int
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[QualityIssue]
    total_rows: int
    valid_rows: int
    quality_score: float  # 0.0 to 1.0
    summary: Dict[str, int]


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    dataset_name: str
    validation_time: datetime
    total_records: int
    valid_records: int
    quality_score: float
    issues_by_type: Dict[QualityIssueType, int]
    issues_by_severity: Dict[str, int]
    field_quality: Dict[str, float]
    recommendations: List[str]
    cleaned_data: Optional[pd.DataFrame] = None


class DataQualityChecker:
    """Comprehensive data quality checking and cleaning system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.quality_thresholds = {
            'z_score_threshold': self.config.get('z_score_threshold', 3.0),
            'iqr_multiplier': self.config.get('iqr_multiplier', 1.5),
            'missing_threshold': self.config.get('missing_threshold', 0.1),  # 10%
            'duplicate_threshold': self.config.get('duplicate_threshold', 0.05),  # 5%
            'quality_score_threshold': self.config.get('quality_score_threshold', 0.8)
        }
        
        # Field-specific validation rules
        self.field_rules = {
            'price': {'min': 0, 'max': float('inf'), 'type': float},
            'volume': {'min': 0, 'max': float('inf'), 'type': float},
            'timestamp': {'type': datetime},
            'symbol': {'type': str, 'min_length': 1, 'max_length': 20}
        }
    
    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str = "unknown") -> QualityReport:
        """Perform comprehensive validation on a DataFrame"""
        logger.info(f"Starting quality validation for dataset: {dataset_name}")
        
        start_time = datetime.now()
        original_count = len(df)
        issues = []
        
        # 1. Check for missing values
        missing_issues = self._check_missing_values(df)
        issues.extend(missing_issues)
        
        # 2. Detect outliers
        outlier_issues = self._detect_outliers(df)
        issues.extend(outlier_issues)
        
        # 3. Check for duplicates
        duplicate_issues = self._check_duplicates(df)
        issues.extend(duplicate_issues)
        
        # 4. Validate data consistency
        consistency_issues = self._check_consistency(df)
        issues.extend(consistency_issues)
        
        # 5. Format validation
        format_issues = self._validate_formats(df)
        issues.extend(format_issues)
        
        # 6. Range validation
        range_issues = self._validate_ranges(df)
        issues.extend(range_issues)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(df, issues)
        valid_records = original_count - len([i for i in issues if i.severity in ['high', 'critical']])
        
        # Group issues by type and severity
        issues_by_type = {}
        issues_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for issue in issues:
            issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1
            issues_by_severity[issue.severity] += 1
        
        # Calculate field-level quality scores
        field_quality = self._calculate_field_quality(df, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, quality_score)
        
        # Create cleaned dataset
        cleaned_data = self._clean_data(df, issues)
        
        report = QualityReport(
            dataset_name=dataset_name,
            validation_time=start_time,
            total_records=original_count,
            valid_records=valid_records,
            quality_score=quality_score,
            issues_by_type=issues_by_type,
            issues_by_severity=issues_by_severity,
            field_quality=field_quality,
            recommendations=recommendations,
            cleaned_data=cleaned_data
        )
        
        logger.info(f"Quality validation completed. Score: {quality_score:.3f}, Issues: {len(issues)}")
        return report
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for missing values and assess impact"""
        issues = []
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_ratio = missing_count / len(df)
            
            if missing_count > 0:
                severity = self._get_missing_severity(missing_ratio)
                
                # Find specific missing value locations
                missing_indices = df[df[column].isna()].index.tolist()[:10]  # Limit to first 10
                
                for idx in missing_indices:
                    issue = QualityIssue(
                        issue_type=QualityIssueType.MISSING_VALUE,
                        field=column,
                        value=None,
                        row_index=idx,
                        description=f"Missing value in {column}. {missing_count}/{len(df)} total missing ({missing_ratio:.1%})",
                        severity=severity,
                        timestamp=datetime.now(),
                        suggestion=self._get_missing_suggestion(column, missing_ratio)
                    )
                    issues.append(issue)
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect outliers using IQR and Z-score methods"""
        issues = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in df.columns:
                # IQR method
                iqr_outliers = self._detect_iqr_outliers(df[column], column)
                issues.extend(iqr_outliers)
                
                # Z-score method
                zscore_outliers = self._detect_zscore_outliers(df[column], column)
                issues.extend(zscore_outliers)
        
        return issues
    
    def _detect_iqr_outliers(self, series: pd.Series, column_name: str) -> List[QualityIssue]:
        """Detect outliers using Interquartile Range method"""
        issues = []
        
        if len(series.dropna()) < 4:  # Need at least 4 values for IQR
            return issues
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.quality_thresholds['iqr_multiplier'] * IQR
        upper_bound = Q3 + self.quality_thresholds['iqr_multiplier'] * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = series[outlier_mask].index.tolist()
        
        for idx in outlier_indices:
            value = series.iloc[idx]
            severity = self._get_outlier_severity(value, series)
            
            issue = QualityIssue(
                issue_type=QualityIssueType.OUTLIER,
                field=column_name,
                value=value,
                row_index=idx,
                description=f"IQR outlier in {column_name}: {value} (bounds: {lower_bound:.2f} - {upper_bound:.2f})",
                severity=severity,
                timestamp=datetime.now(),
                suggestion=f"Consider removing or investigating value {value}"
            )
            issues.append(issue)
        
        return issues
    
    def _detect_zscore_outliers(self, series: pd.Series, column_name: str) -> List[QualityIssue]:
        """Detect outliers using Z-score method"""
        issues = []
        
        if len(series.dropna()) < 2:
            return issues
        
        mean_val = series.mean()
        std_val = series.std()
        
        if std_val == 0:  # No variation
            return issues
        
        z_scores = np.abs((series - mean_val) / std_val)
        outlier_mask = z_scores > self.quality_thresholds['z_score_threshold']
        outlier_indices = series[outlier_mask].index.tolist()
        
        for idx in outlier_indices:
            value = series.iloc[idx]
            z_score = z_scores.iloc[idx]
            severity = self._get_outlier_severity(value, series)
            
            issue = QualityIssue(
                issue_type=QualityIssueType.OUTLIER,
                field=column_name,
                value=value,
                row_index=idx,
                description=f"Z-score outlier in {column_name}: {value} (z-score: {z_score:.2f})",
                severity=severity,
                timestamp=datetime.now(),
                suggestion=f"Z-score {z_score:.2f} exceeds threshold {self.quality_thresholds['z_score_threshold']}"
            )
            issues.append(issue)
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for duplicate records"""
        issues = []
        
        # Full row duplicates
        duplicate_mask = df.duplicated(keep=False)
        duplicate_indices = df[duplicate_mask].index.tolist()
        
        if len(duplicate_indices) > 0:
            duplicate_ratio = len(duplicate_indices) / len(df)
            severity = 'high' if duplicate_ratio > 0.1 else 'medium'
            
            for idx in duplicate_indices[:20]:  # Limit to first 20
                issue = QualityIssue(
                    issue_type=QualityIssueType.DUPLICATE,
                    field="all_fields",
                    value="duplicate_row",
                    row_index=idx,
                    description=f"Duplicate row found. Total duplicates: {len(duplicate_indices)} ({duplicate_ratio:.1%})",
                    severity=severity,
                    timestamp=datetime.now(),
                    suggestion="Remove duplicate records to avoid data skewing"
                )
                issues.append(issue)
        
        return issues
    
    def _check_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for data consistency issues"""
        issues = []
        
        # Check OHLC consistency (if applicable)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            consistency_issues = self._check_ohlc_consistency(df)
            issues.extend(consistency_issues)
        
        # Check timestamp consistency
        if 'timestamp' in df.columns:
            timestamp_issues = self._check_timestamp_consistency(df)
            issues.extend(timestamp_issues)
        
        # Check volume consistency
        if 'volume' in df.columns:
            volume_issues = self._check_volume_consistency(df)
            issues.extend(volume_issues)
        
        return issues
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check OHLC price consistency"""
        issues = []
        
        for idx, row in df.iterrows():
            try:
                open_price = float(row['open'])
                high_price = float(row['high'])
                low_price = float(row['low'])
                close_price = float(row['close'])
                
                # High should be the highest
                if high_price < max(open_price, close_price, low_price):
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT,
                        field="high",
                        value=high_price,
                        row_index=idx,
                        description=f"High price {high_price} is not the highest (O:{open_price}, L:{low_price}, C:{close_price})",
                        severity="high",
                        timestamp=datetime.now(),
                        suggestion="Verify price data integrity"
                    )
                    issues.append(issue)
                
                # Low should be the lowest
                if low_price > min(open_price, high_price, close_price):
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT,
                        field="low",
                        value=low_price,
                        row_index=idx,
                        description=f"Low price {low_price} is not the lowest (O:{open_price}, H:{high_price}, C:{close_price})",
                        severity="high",
                        timestamp=datetime.now(),
                        suggestion="Verify price data integrity"
                    )
                    issues.append(issue)
                    
            except (ValueError, TypeError):
                # Handle non-numeric values
                pass
        
        return issues
    
    def _check_timestamp_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check timestamp ordering and gaps"""
        issues = []
        
        if len(df) < 2:
            return issues
        
        # Sort by timestamp to check ordering
        if df['timestamp'].dtype == 'object':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                return issues
        
        # Check for non-chronological order
        timestamp_diff = df['timestamp'].diff()
        negative_diff_indices = timestamp_diff[timestamp_diff < timedelta(0)].index.tolist()
        
        for idx in negative_diff_indices:
            issue = QualityIssue(
                issue_type=QualityIssueType.INCONSISTENT,
                field="timestamp",
                value=df.loc[idx, 'timestamp'],
                row_index=idx,
                description=f"Timestamp out of order: {df.loc[idx, 'timestamp']}",
                severity="medium",
                timestamp=datetime.now(),
                suggestion="Sort data by timestamp or check data source"
            )
            issues.append(issue)
        
        return issues
    
    def _check_volume_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check volume data for anomalies"""
        issues = []
        
        if 'volume' not in df.columns:
            return issues
        
        # Check for negative volumes
        negative_volume_mask = df['volume'] < 0
        negative_indices = df[negative_volume_mask].index.tolist()
        
        for idx in negative_indices:
            issue = QualityIssue(
                issue_type=QualityIssueType.INCONSISTENT,
                field="volume",
                value=df.loc[idx, 'volume'],
                row_index=idx,
                description=f"Negative volume: {df.loc[idx, 'volume']}",
                severity="critical",
                timestamp=datetime.now(),
                suggestion="Volume cannot be negative - check data source"
            )
            issues.append(issue)
        
        return issues
    
    def _validate_formats(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Validate data formats and types"""
        issues = []
        
        for field, rules in self.field_rules.items():
            if field in df.columns:
                field_issues = self._validate_field_format(df[field], field, rules)
                issues.extend(field_issues)
        
        return issues
    
    def _validate_field_format(self, series: pd.Series, field_name: str, rules: Dict) -> List[QualityIssue]:
        """Validate specific field format"""
        issues = []
        
        expected_type = rules.get('type')
        if expected_type:
            for idx, value in series.items():
                if pd.isna(value):
                    continue
                
                if expected_type == str and not isinstance(value, str):
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INVALID_FORMAT,
                        field=field_name,
                        value=value,
                        row_index=idx,
                        description=f"Expected string, got {type(value).__name__}: {value}",
                        severity="medium",
                        timestamp=datetime.now(),
                        suggestion=f"Convert {field_name} to string format"
                    )
                    issues.append(issue)
                
                elif expected_type in [int, float] and not isinstance(value, (int, float)):
                    try:
                        float(value)  # Try to convert
                    except (ValueError, TypeError):
                        issue = QualityIssue(
                            issue_type=QualityIssueType.INVALID_FORMAT,
                            field=field_name,
                            value=value,
                            row_index=idx,
                            description=f"Expected numeric, got {type(value).__name__}: {value}",
                            severity="high",
                            timestamp=datetime.now(),
                            suggestion=f"Convert {field_name} to numeric format"
                        )
                        issues.append(issue)
        
        return issues
    
    def _validate_ranges(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Validate that values are within expected ranges"""
        issues = []
        
        for field, rules in self.field_rules.items():
            if field in df.columns:
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None:
                    below_min_mask = df[field] < min_val
                    below_min_indices = df[below_min_mask].index.tolist()
                    
                    for idx in below_min_indices:
                        issue = QualityIssue(
                            issue_type=QualityIssueType.OUT_OF_RANGE,
                            field=field,
                            value=df.loc[idx, field],
                            row_index=idx,
                            description=f"Value {df.loc[idx, field]} below minimum {min_val}",
                            severity="high",
                            timestamp=datetime.now(),
                            suggestion=f"Ensure {field} values are >= {min_val}"
                        )
                        issues.append(issue)
                
                if max_val is not None and max_val != float('inf'):
                    above_max_mask = df[field] > max_val
                    above_max_indices = df[above_max_mask].index.tolist()
                    
                    for idx in above_max_indices:
                        issue = QualityIssue(
                            issue_type=QualityIssueType.OUT_OF_RANGE,
                            field=field,
                            value=df.loc[idx, field],
                            row_index=idx,
                            description=f"Value {df.loc[idx, field]} above maximum {max_val}",
                            severity="high",
                            timestamp=datetime.now(),
                            suggestion=f"Ensure {field} values are <= {max_val}"
                        )
                        issues.append(issue)
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Calculate overall data quality score (0.0 to 1.0)"""
        if len(df) == 0:
            return 0.0
        
        # Weight issues by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0}
        total_penalty = 0.0
        
        for issue in issues:
            penalty = severity_weights.get(issue.severity, 0.5)
            total_penalty += penalty
        
        # Normalize by dataset size
        max_possible_penalty = len(df) * 1.0  # Assuming worst case: all critical issues
        penalty_ratio = min(total_penalty / max_possible_penalty, 1.0) if max_possible_penalty > 0 else 0.0
        
        quality_score = max(1.0 - penalty_ratio, 0.0)
        return round(quality_score, 3)
    
    def _calculate_field_quality(self, df: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
        """Calculate quality score for each field"""
        field_quality = {}
        
        for column in df.columns:
            field_issues = [issue for issue in issues if issue.field == column]
            field_penalties = sum(
                {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0}.get(issue.severity, 0.5)
                for issue in field_issues
            )
            
            max_penalty = len(df) * 1.0
            penalty_ratio = min(field_penalties / max_penalty, 1.0) if max_penalty > 0 else 0.0
            field_quality[column] = max(1.0 - penalty_ratio, 0.0)
        
        return field_quality
    
    def _generate_recommendations(self, issues: List[QualityIssue], quality_score: float) -> List[str]:
        """Generate actionable recommendations based on issues found"""
        recommendations = []
        
        issue_counts = {}
        for issue in issues:
            key = f"{issue.issue_type.value}_{issue.severity}"
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        # Missing value recommendations
        if any('missing_value' in key for key in issue_counts):
            recommendations.append("Consider implementing data imputation strategies for missing values")
        
        # Outlier recommendations
        if any('outlier' in key for key in issue_counts):
            recommendations.append("Review outlier detection thresholds and investigate extreme values")
        
        # Duplicate recommendations
        if any('duplicate' in key for key in issue_counts):
            recommendations.append("Implement deduplication process in data pipeline")
        
        # Quality score recommendations
        if quality_score < 0.7:
            recommendations.append("Data quality is below acceptable threshold - immediate action required")
        elif quality_score < 0.9:
            recommendations.append("Data quality is moderate - consider implementing additional validation rules")
        
        return recommendations
    
    def _clean_data(self, df: pd.DataFrame, issues: List[QualityIssue]) -> pd.DataFrame:
        """Create cleaned version of the data"""
        cleaned_df = df.copy()
        
        # Remove critical issues
        critical_indices = {issue.row_index for issue in issues if issue.severity == 'critical'}
        if critical_indices:
            cleaned_df = cleaned_df.drop(index=list(critical_indices))
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values (simple forward fill for now)
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].ffill()
        
        return cleaned_df
    
    def _get_missing_severity(self, missing_ratio: float) -> str:
        """Determine severity of missing values based on ratio"""
        if missing_ratio > 0.5:
            return 'critical'
        elif missing_ratio > 0.2:
            return 'high'
        elif missing_ratio > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _get_outlier_severity(self, value: float, series: pd.Series) -> str:
        """Determine outlier severity based on deviation"""
        median = series.median()
        mad = np.median(np.abs(series - median))  # Median Absolute Deviation
        
        if mad == 0:
            return 'low'
        
        deviation_ratio = abs(value - median) / mad
        
        if deviation_ratio > 10:
            return 'critical'
        elif deviation_ratio > 5:
            return 'high'
        elif deviation_ratio > 3:
            return 'medium'
        else:
            return 'low'
    
    def _get_missing_suggestion(self, field: str, missing_ratio: float) -> str:
        """Get suggestion for handling missing values"""
        if missing_ratio > 0.5:
            return f"Consider excluding {field} from analysis due to high missing rate"
        elif missing_ratio > 0.2:
            return f"Implement robust imputation strategy for {field}"
        else:
            return f"Apply forward fill or interpolation for {field}"

    def interpolate_missing_values(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing values using specified method"""
        interpolated_df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if df[column].isna().any():
                if method == 'linear':
                    interpolated_df[column] = df[column].interpolate(method='linear')
                elif method == 'forward_fill':
                    interpolated_df[column] = df[column].fillna(method='ffill')
                elif method == 'backward_fill':
                    interpolated_df[column] = df[column].fillna(method='bfill')
                elif method == 'mean':
                    interpolated_df[column] = df[column].fillna(df[column].mean())
                elif method == 'median':
                    interpolated_df[column] = df[column].fillna(df[column].median())
        
        logger.info(f"Interpolated missing values using {method} method")
        return interpolated_df

    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the dataset"""
        cleaned_df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                cleaned_df = cleaned_df[z_scores <= 3]
        
        logger.info(f"Removed outliers using {method} method. Removed {len(df) - len(cleaned_df)} rows")
        return cleaned_df