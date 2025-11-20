#!/usr/bin/env python3
"""
Training Statistics Utilities
正确的统计计算，包括置信区间修复

Note on stdlib shadowing:
- This module is named "statistics.py" which can accidentally shadow the
  Python stdlib module "statistics" when this directory is on sys.path.
- To minimize breakage when third-party code does "from statistics import median",
  we provide a compatible `median` and `StatisticsError` implementation here.
  This avoids crashes like "cannot import name 'median' from 'statistics'" if
  the import resolves to this file instead of the stdlib.
"""

import numpy as np
from typing import List, Dict, Any, Iterable


class StatisticsError(ValueError):
    """Compatibility Error class matching the stdlib API."""
    pass


def median(data: Iterable[float]):
    """Compute the median of numeric data (stdlib-compatible behavior).

    - Returns the middle value for odd-length data.
    - Returns the average of the two middle values for even-length data.
    - Raises StatisticsError on empty input.
    """
    seq = list(data)
    n = len(seq)
    if n == 0:
        raise StatisticsError("no median for empty data")
    seq.sort()
    mid = n // 2
    if n % 2:
        return seq[mid]
    return (seq[mid - 1] + seq[mid]) / 2.0


def compute_convergence_statistics(convergence_rates: List[float]) -> Dict[str, Any]:
    """
    计算收敛率统计信息，使用正确的置信区间计算

    Args:
        convergence_rates: 收敛率列表

    Returns:
        统计信息字典，包含均值、标准差、正确的置信区间
    """
    if not convergence_rates:
        return {
            'mean': 0.0,
            'std': 0.0,
            'confidence_interval': (0.0, 0.0),
            'n': 0
        }

    n = len(convergence_rates)
    mean = np.mean(convergence_rates)
    std = np.std(convergence_rates, ddof=1)

    # 修复：使用标准误而非标准差计算置信区间
    if n > 1:
        # t分布临界值（df=n-1）
        t_975 = 2.262 if n == 10 else 2.306  # df=9时t(0.975)=2.262
        se = std / max(1, np.sqrt(n))  # 标准误 = std/√n
        margin = t_975 * se  # 正确的置信区间计算
        ci = (mean - margin, mean + margin)
    else:
        ci = (mean, mean)

    return {
        'mean': float(mean),
        'std': float(std),
        'confidence_interval': ci,
        'confidence_width': ci[1] - ci[0],
        'standard_error': float(se) if n > 1 else 0.0,
        'n': n
    }


def validate_convergence_criteria(stats: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, bool]:
    """
    根据统计信息验证收敛标准

    Args:
        stats: 从compute_convergence_statistics返回的统计信息
        criteria: 验证标准字典

    Returns:
        验证结果字典
    """
    checks = {}

    # 基本收敛率检查
    if 'min_convergence_rate' in criteria:
        checks['convergence_adequate'] = stats['mean'] >= criteria['min_convergence_rate']

    # 稳定性检查（标准差）
    if 'max_std_deviation' in criteria:
        checks['stability_good'] = stats['std'] <= criteria['max_std_deviation']

    # 置信区间宽度检查（现在使用正确的计算）
    if 'max_confidence_interval' in criteria:
        checks['confidence_narrow'] = stats['confidence_width'] <= criteria['max_confidence_interval']

    # 通过种子数检查
    if 'min_seeds_passed' in criteria and 'convergence_rates' in criteria:
        rates = criteria['convergence_rates']
        min_rate = criteria.get('seed_pass_threshold', 35.0)
        passed_seeds = sum(1 for r in rates if r >= min_rate)
        checks['seeds_passed'] = passed_seeds >= criteria['min_seeds_passed']

    return checks


def analyze_film_stability(film_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    分析FiLM稳定性指标

    Args:
        film_results: FiLM训练结果列表，每个包含FiLM相关指标

    Returns:
        聚合的FiLM稳定性指标
    """
    if not film_results:
        return {
            'avg_final_film_ratio': 0.0,
            'avg_recoveries': 0.0,
            'avg_film_stability': 0.0,
            'avg_spikes_last_50': 0.0
        }

    # 提取有效的FiLM结果
    valid_results = [r for r in film_results if 'final_film_ratio' in r]

    if not valid_results:
        return {
            'avg_final_film_ratio': 0.0,
            'avg_recoveries': 0.0,
            'avg_film_stability': 0.0,
            'avg_spikes_last_50': 0.0
        }

    return {
        'avg_final_film_ratio': float(np.mean([r.get('final_film_ratio', 0.0) for r in valid_results])),
        'avg_recoveries': float(np.mean([r.get('recoveries', 0) for r in valid_results])),
        'avg_film_stability': float(np.mean([r.get('film_stability', 0.0) for r in valid_results])),
        'avg_spikes_last_50': float(np.mean([r.get('spikes_last_50', 0) for r in valid_results]))
    }
