#!/usr/bin/env python3
"""
GAIA Benchmark Results Analysis Tool

This script provides comprehensive analysis of GAIA benchmark results stored in JSONL format.
It analyzes performance metrics, error patterns, agent behaviors, and provides detailed insights.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re


class GAIAResultsAnalyzer:
    """Comprehensive analyzer for GAIA benchmark results"""
    
    def __init__(self, output_dir: str = "./output/validation"):
        self.output_dir = Path(output_dir)
        self.results = []
        self.df = None
        self.metadata = None
        
    def load_results(self, file_pattern: str = "*.jsonl") -> None:
        """Load all JSONL result files matching the pattern"""
        jsonl_files = list(self.output_dir.glob(file_pattern))
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {self.output_dir}")
        
        print(f"Found {len(jsonl_files)} JSONL files:")
        for file in jsonl_files:
            print(f"  - {file.name}")
        
        all_results = []
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as f:
                file_results = []
                for line_num, line in enumerate(f, 1):
                    try:
                        result = json.loads(line.strip())
                        result['source_file'] = file.name
                        file_results.append(result)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON in {file.name}:{line_num} - {e}")
                
                all_results.extend(file_results)
                print(f"Loaded {len(file_results)} results from {file.name}")
        
        self.results = all_results
        self.df = pd.DataFrame(all_results)
        print(f"Total results loaded: {len(self.results)}")
        
        # Load metadata for level information
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load GAIA metadata file for level information"""
        # Look for metadata.jsonl in common GAIA data directories
        possible_metadata_paths = [
            Path("./data/gaia/validation/metadata.jsonl"),
            Path("./OAgents/example/oagents_deep_research/data/gaia/validation/metadata.jsonl"),
            Path("../data/gaia/validation/metadata.jsonl"),
            self.output_dir / "../data/gaia/validation/metadata.jsonl",
            self.output_dir / "../../data/gaia/validation/metadata.jsonl"
        ]
        
        metadata_path = None
        for path in possible_metadata_paths:
            if path.exists():
                metadata_path = path
                break
        
        if metadata_path is None:
            print("Warning: Could not find metadata.jsonl file. Level analysis will be limited.")
            self.metadata = {}
            return
        
        try:
            metadata_dict = {}
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        metadata = json.loads(line.strip())
                        task_id = metadata.get('task_id')
                        level = metadata.get('Level')
                        if task_id and level is not None:
                            metadata_dict[task_id] = {
                                'level': level,
                                'question': metadata.get('Question', ''),
                                'final_answer': metadata.get('Final answer', '')
                            }
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON in metadata.jsonl:{line_num} - {e}")
            
            self.metadata = metadata_dict
            print(f"Loaded metadata for {len(metadata_dict)} tasks from {metadata_path}")
            
            # Add level information to DataFrame
            if self.df is not None and not self.df.empty:
                self.df['gaia_level'] = self.df['task_id'].map(
                    lambda x: self.metadata.get(x, {}).get('level', 'unknown')
                )
                print(f"Mapped level information to {(self.df['gaia_level'] != 'unknown').sum()} tasks")
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def calculate_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        
        # Exact match accuracy (case-insensitive)
        correct_exact = sum(1 for r in self.results 
                           if r.get('prediction', '').lower().strip() == r.get('true_answer', '').lower().strip())
        
        # Successful completion (no errors)
        successful = sum(1 for r in self.results 
                        if not r.get('agent_error') and not r.get('parsing_error'))
        
        # Cases where iteration limit was exceeded
        iteration_exceeded = sum(1 for r in self.results if r.get('iteration_limit_exceeded', False))
        
        return {
            'exact_match_accuracy': correct_exact / total if total > 0 else 0,
            'success_rate': successful / total if total > 0 else 0,
            'iteration_limit_exceeded_rate': iteration_exceeded / total if total > 0 else 0,
            'total_tasks': total,
            'correct_answers': correct_exact,
            'successful_runs': successful
        }
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns and types"""
        if not self.results:
            return {}
        
        agent_errors = [r for r in self.results if r.get('agent_error')]
        parsing_errors = [r for r in self.results if r.get('parsing_error')]
        iteration_exceeded = [r for r in self.results if r.get('iteration_limit_exceeded')]
        
        # Categorize agent errors
        error_categories = defaultdict(list)
        for result in agent_errors:
            error_msg = result['agent_error']
            if 'timeout' in error_msg.lower():
                error_categories['timeout'].append(result)
            elif 'rate limit' in error_msg.lower():
                error_categories['rate_limit'].append(result)
            elif 'connection' in error_msg.lower():
                error_categories['connection'].append(result)
            elif 'api' in error_msg.lower():
                error_categories['api_error'].append(result)
            else:
                error_categories['other'].append(result)
        
        return {
            'total_agent_errors': len(agent_errors),
            'total_parsing_errors': len(parsing_errors),
            'total_iteration_exceeded': len(iteration_exceeded),
            'error_categories': dict(error_categories),
            'error_rates': {
                'agent_error_rate': len(agent_errors) / len(self.results),
                'parsing_error_rate': len(parsing_errors) / len(self.results),
                'iteration_exceeded_rate': len(iteration_exceeded) / len(self.results)
            }
        }
    
    def analyze_performance_by_task_type(self) -> Dict[str, Any]:
        """Analyze performance by GAIA task type/level"""
        if self.df is None or self.df.empty:
            return {}
        
        # Use gaia_level column if available, otherwise try to extract from task_id
        if 'gaia_level' not in self.df.columns:
            self.df['gaia_level'] = self.df['task_id'].apply(
                lambda x: re.search(r'level(\d+)', str(x)).group(1) if re.search(r'level(\d+)', str(x)) else 'unknown'
            )
        
        performance_by_level = {}
        for level in self.df['gaia_level'].unique():
            level_results = self.df[self.df['gaia_level'] == level]
            correct = sum(level_results['prediction'].str.lower().str.strip() == 
                         level_results['true_answer'].str.lower().str.strip())
            
            performance_by_level[f'level_{level}'] = {
                'total_tasks': len(level_results),
                'correct_answers': correct,
                'accuracy': correct / len(level_results) if len(level_results) > 0 else 0,
                'avg_duration': self._calculate_avg_duration(level_results),
                'error_rate': sum(level_results['agent_error'].notna()) / len(level_results)
            }
        
        return performance_by_level
    
    def analyze_task_level_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of GAIA problem levels"""
        if self.df is None or self.df.empty:
            return {}
        
        # Use gaia_level column if available, otherwise try to extract from task_id
        if 'gaia_level' not in self.df.columns:
            self.df['gaia_level'] = self.df['task_id'].apply(
                lambda x: re.search(r'level(\d+)', str(x)).group(1) if re.search(r'level(\d+)', str(x)) else 'unknown'
            )
        
        # Count distribution
        level_counts = self.df['gaia_level'].value_counts()
        
        # Sort by level number for better display
        try:
            level_counts = level_counts.reindex(
                sorted(level_counts.index, key=lambda x: int(x) if str(x).isdigit() else 999)
            )
        except:
            level_counts = level_counts.sort_index()
        
        total_tasks = len(self.df)
        
        distribution = {}
        for level in level_counts.index:
            count = level_counts[level]
            distribution[f'level_{level}'] = {
                'count': int(count),
                'percentage': (count / total_tasks) * 100,
                'proportion': count / total_tasks
            }
        
        return {
            'total_tasks': total_tasks,
            'level_distribution': distribution,
            'unique_levels': len(level_counts),
            'most_common_level': level_counts.index[0] if len(level_counts) > 0 else 'unknown',
            'least_common_level': level_counts.index[-1] if len(level_counts) > 0 else 'unknown'
        }
    
    def analyze_agent_behavior(self) -> Dict[str, Any]:
        """Analyze agent reasoning and tool usage patterns"""
        if not self.results:
            return {}

        # Initialize counters
        total_steps = []
        search_actions = []
        reasoning_quality = []

        # Step type analysis
        task_steps_count = []
        planning_steps_count = []
        action_steps_count = []

        # Tool usage analysis
        tool_usage = Counter()
        tool_calls_per_task = []

        # Timing analysis for steps
        step_durations = []

        # LLM call analysis
        llm_calls_per_task = []
        llm_call_statistics = {
            'planning_calls': 0,
            'action_calls': 0,
            'search_agent_calls': 0,
            'total_llm_calls': 0,
            'calls_by_task': []
        }

        # Planning analysis
        planning_patterns = {
            'has_facts': 0,
            'has_detailed_plan': 0,
            'avg_facts_length': [],
            'avg_plan_length': []
        }

        # Action analysis
        action_patterns = {
            'successful_actions': 0,
            'failed_actions': 0,
            'python_code_executions': 0,
            'search_agent_calls': 0,
            'file_inspections': 0
        }
        
        for result in self.results:
            steps = result.get('intermediate_steps', [])
            total_steps.append(len(steps))

            # Initialize LLM call counter for this task
            task_llm_calls = 0

            # Analyze search agent actions
            search_records = result.get('search_agent_actions', [])
            search_actions.append(len(search_records))

            # Count LLM calls from search agent
            if search_records:
                # Each search agent action typically involves 1 LLM call
                search_agent_llm_calls = len(search_records)
                task_llm_calls += search_agent_llm_calls
                llm_call_statistics['search_agent_calls'] += search_agent_llm_calls

            # Count steps by type
            task_steps = [s for s in steps if s.get('step_type') == 'task']
            planning_steps = [s for s in steps if s.get('step_type') == 'planning']
            action_steps = [s for s in steps if s.get('step_type') == 'action']

            task_steps_count.append(len(task_steps))
            planning_steps_count.append(len(planning_steps))
            action_steps_count.append(len(action_steps))
            
            # Analyze planning steps
            if planning_steps:
                # Each planning step typically involves 1 LLM call
                planning_llm_calls = len(planning_steps)
                task_llm_calls += planning_llm_calls
                llm_call_statistics['planning_calls'] += planning_llm_calls

                for planning_step in planning_steps:
                    facts = planning_step.get('facts', '')
                    plan = planning_step.get('plan', '')

                    if facts:
                        planning_patterns['has_facts'] += 1
                        planning_patterns['avg_facts_length'].append(len(facts.split()))

                    if plan:
                        planning_patterns['has_detailed_plan'] += 1
                        planning_patterns['avg_plan_length'].append(len(plan.split()))
                        reasoning_quality.append(len(plan.split()))
                    else:
                        reasoning_quality.append(0)
            else:
                reasoning_quality.append(0)
            
            # Analyze action steps
            task_tool_calls = 0
            for action_step in action_steps:
                # Each action step typically involves 1 LLM call
                task_llm_calls += 1
                llm_call_statistics['action_calls'] += 1

                # Check for errors
                if action_step.get('error'):
                    action_patterns['failed_actions'] += 1
                else:
                    action_patterns['successful_actions'] += 1

                # Analyze tool calls
                tool_calls = action_step.get('tool_calls', [])
                task_tool_calls += len(tool_calls)

                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_name = tool_call.get('function', {}).get('name', 'unknown')
                        tool_usage[function_name] += 1

                        # Categorize tool types
                        if function_name == 'python_interpreter':
                            action_patterns['python_code_executions'] += 1

                            # Check if python code contains search_agent calls
                            arguments = tool_call.get('function', {}).get('arguments', '')
                            if 'search_agent(' in arguments:
                                action_patterns['search_agent_calls'] += 1
                            if 'inspect_file' in arguments:
                                action_patterns['file_inspections'] += 1

                        elif 'search_agent' in function_name:
                            action_patterns['search_agent_calls'] += 1
                        elif 'inspect_file' in function_name:
                            action_patterns['file_inspections'] += 1

                # Analyze step duration
                start_time = action_step.get('start_time')
                end_time = action_step.get('end_time')
                if start_time and end_time:
                    duration = end_time - start_time
                    step_durations.append(duration)

            tool_calls_per_task.append(task_tool_calls)

            # Record total LLM calls for this task
            llm_calls_per_task.append(task_llm_calls)
            llm_call_statistics['total_llm_calls'] += task_llm_calls
            llm_call_statistics['calls_by_task'].append({
                'task_id': result.get('task_id', 'unknown'),
                'llm_calls': task_llm_calls,
                'level': result.get('task', 'unknown')
            })
        
        return {
            'step_statistics': {
                'avg_steps': np.mean(total_steps) if total_steps else 0,
                'median_steps': np.median(total_steps) if total_steps else 0,
                'max_steps': max(total_steps) if total_steps else 0,
                'min_steps': min(total_steps) if total_steps else 0,
                'avg_task_steps': np.mean(task_steps_count) if task_steps_count else 0,
                'avg_planning_steps': np.mean(planning_steps_count) if planning_steps_count else 0,
                'avg_action_steps': np.mean(action_steps_count) if action_steps_count else 0
            },
            'llm_call_statistics': {
                'avg_llm_calls_per_task': np.mean(llm_calls_per_task) if llm_calls_per_task else 0,
                'median_llm_calls_per_task': np.median(llm_calls_per_task) if llm_calls_per_task else 0,
                'max_llm_calls_per_task': max(llm_calls_per_task) if llm_calls_per_task else 0,
                'min_llm_calls_per_task': min(llm_calls_per_task) if llm_calls_per_task else 0,
                'total_llm_calls': llm_call_statistics['total_llm_calls'],
                'planning_calls': llm_call_statistics['planning_calls'],
                'action_calls': llm_call_statistics['action_calls'],
                'search_agent_calls': llm_call_statistics['search_agent_calls'],
                'calls_by_level': self._analyze_llm_calls_by_level(llm_call_statistics['calls_by_task']),
                'tasks_by_llm_usage': self._categorize_tasks_by_llm_usage(llm_calls_per_task)
            },
            'search_statistics': {
                'avg_searches': np.mean(search_actions) if search_actions else 0,
                'median_searches': np.median(search_actions) if search_actions else 0,
                'tasks_with_search': sum(1 for x in search_actions if x > 0)
            },
            'reasoning_statistics': {
                'avg_plan_words': np.mean(reasoning_quality) if reasoning_quality else 0,
                'tasks_with_planning': sum(1 for x in reasoning_quality if x > 0),
                'tasks_with_facts': planning_patterns['has_facts'],
                'tasks_with_detailed_plan': planning_patterns['has_detailed_plan'],
                'avg_facts_words': np.mean(planning_patterns['avg_facts_length']) if planning_patterns['avg_facts_length'] else 0,
                'avg_plan_words_detailed': np.mean(planning_patterns['avg_plan_length']) if planning_patterns['avg_plan_length'] else 0
            },
            'tool_usage_statistics': {
                'most_used_tools': dict(tool_usage.most_common(10)),
                'avg_tool_calls_per_task': np.mean(tool_calls_per_task) if tool_calls_per_task else 0,
                'python_executions': action_patterns['python_code_executions'],
                'search_agent_calls': action_patterns['search_agent_calls'],
                'file_inspections': action_patterns['file_inspections']
            },
            'action_statistics': {
                'successful_actions': action_patterns['successful_actions'],
                'failed_actions': action_patterns['failed_actions'],
                'success_rate': action_patterns['successful_actions'] / (action_patterns['successful_actions'] + action_patterns['failed_actions']) if (action_patterns['successful_actions'] + action_patterns['failed_actions']) > 0 else 0,
                'avg_step_duration': np.mean(step_durations) if step_durations else 0
            }
        }
    
    def analyze_timing_performance(self) -> Dict[str, Any]:
        """Analyze task execution timing"""
        if not self.results:
            return {}
        
        durations = []
        for result in self.results:
            start_time = result.get('start_time')
            end_time = result.get('end_time')
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    durations.append(duration)
                except:
                    continue
        
        if not durations:
            return {"error": "No valid timing data found"}
        
        return {
            'avg_duration_seconds': np.mean(durations),
            'median_duration_seconds': np.median(durations),
            'max_duration_seconds': max(durations),
            'min_duration_seconds': min(durations),
            'std_duration_seconds': np.std(durations),
            'total_runtime_seconds': sum(durations),
            'duration_analysis_by_level': self._analyze_duration_by_level()
        }
    
    def generate_detailed_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.results:
            return "No results loaded"
        
        accuracy_metrics = self.calculate_accuracy()
        error_analysis = self.analyze_errors()
        task_performance = self.analyze_performance_by_task_type()
        level_distribution = self.analyze_task_level_distribution()
        agent_behavior = self.analyze_agent_behavior()
        timing_analysis = self.analyze_timing_performance()
        step_patterns = self.analyze_detailed_step_patterns()
        code_patterns = self.analyze_python_code_patterns()
        consistency_analysis = self.analyze_consistency_across_runs()
        
        report = []
        report.append("=" * 80)
        report.append("GAIA BENCHMARK RESULTS ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Overall Performance
        report.append("📊 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Tasks: {accuracy_metrics['total_tasks']}")
        report.append(f"Exact Match Accuracy: {accuracy_metrics['exact_match_accuracy']:.2%}")
        report.append(f"Success Rate (No Errors): {accuracy_metrics['success_rate']:.2%}")
        report.append(f"Correct Answers: {accuracy_metrics['correct_answers']}")
        report.append(f"Successful Runs: {accuracy_metrics['successful_runs']}")
        report.append("")
        
        # GAIA Level Distribution
        if level_distribution:
            report.append("📊 GAIA PROBLEM LEVEL DISTRIBUTION")
            report.append("-" * 40)
            report.append(f"Total Tasks: {level_distribution['total_tasks']}")
            report.append(f"Unique Levels: {level_distribution['unique_levels']}")
            report.append(f"Most Common Level: Level {level_distribution['most_common_level']}")
            report.append(f"Least Common Level: Level {level_distribution['least_common_level']}")
            report.append("")
            
            report.append("Level Distribution:")
            # Sort levels for consistent display
            sorted_levels = sorted(level_distribution['level_distribution'].items(), 
                                 key=lambda x: int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 999)
            
            for level_name, stats in sorted_levels:
                level_num = level_name.replace('level_', '')
                report.append(f"  Level {level_num}: {stats['count']} tasks ({stats['percentage']:.1f}%)")
            report.append("")
        
        # Error Analysis
        report.append("❌ ERROR ANALYSIS")
        report.append("-" * 40)
        report.append(f"Agent Errors: {error_analysis['total_agent_errors']} ({error_analysis['error_rates']['agent_error_rate']:.2%})")
        report.append(f"Parsing Errors: {error_analysis['total_parsing_errors']} ({error_analysis['error_rates']['parsing_error_rate']:.2%})")
        report.append(f"Iteration Limit Exceeded: {error_analysis['total_iteration_exceeded']} ({error_analysis['error_rates']['iteration_exceeded_rate']:.2%})")
        report.append("")
        
        if error_analysis['error_categories']:
            report.append("Error Categories:")
            for category, errors in error_analysis['error_categories'].items():
                report.append(f"  - {category.title()}: {len(errors)} cases")
        report.append("")
        
        # Performance by Task Level
        if task_performance:
            report.append("📈 PERFORMANCE BY TASK LEVEL")
            report.append("-" * 40)
            for level, metrics in task_performance.items():
                report.append(f"{level.upper()}:")
                report.append(f"  - Tasks: {metrics['total_tasks']}")
                report.append(f"  - Accuracy: {metrics['accuracy']:.2%}")
                report.append(f"  - Average Duration: {metrics['avg_duration']:.1f}s")
                report.append(f"  - Error Rate: {metrics['error_rate']:.2%}")
            report.append("")
        
        # Agent Behavior Analysis
        report.append("🤖 AGENT BEHAVIOR ANALYSIS")
        report.append("-" * 40)
        step_stats = agent_behavior['step_statistics']
        llm_stats = agent_behavior['llm_call_statistics']
        search_stats = agent_behavior['search_statistics']
        reasoning_stats = agent_behavior['reasoning_statistics']
        tool_stats = agent_behavior['tool_usage_statistics']
        action_stats = agent_behavior['action_statistics']
        
        report.append("📋 Step Analysis:")
        report.append(f"  Average Total Steps per Task: {step_stats['avg_steps']:.1f}")
        report.append(f"  Median Steps per Task: {step_stats['median_steps']:.1f}")
        report.append(f"  Max Steps: {step_stats['max_steps']}")
        report.append(f"  Average Task Steps: {step_stats['avg_task_steps']:.1f}")
        report.append(f"  Average Planning Steps: {step_stats['avg_planning_steps']:.1f}")
        report.append(f"  Average Action Steps: {step_stats['avg_action_steps']:.1f}")
        report.append("")

        report.append("🧠 LLM Call Analysis:")
        report.append(f"  Average LLM Calls per Task: {llm_stats['avg_llm_calls_per_task']:.1f}")
        report.append(f"  Median LLM Calls per Task: {llm_stats['median_llm_calls_per_task']:.1f}")
        report.append(f"  Max LLM Calls per Task: {llm_stats['max_llm_calls_per_task']}")
        report.append(f"  Total LLM Calls: {llm_stats['total_llm_calls']}")
        report.append(f"  Planning Calls: {llm_stats['planning_calls']}")
        report.append(f"  Action Calls: {llm_stats['action_calls']}")
        report.append(f"  Search Agent Calls: {llm_stats['search_agent_calls']}")

        # LLM usage distribution
        usage_stats = llm_stats['tasks_by_llm_usage']
        report.append("  Task Distribution by LLM Usage:")
        report.append(f"    Low (1-5 calls): {usage_stats['low_usage_tasks']} tasks")
        report.append(f"    Medium (6-15 calls): {usage_stats['medium_usage_tasks']} tasks")
        report.append(f"    High (16-30 calls): {usage_stats['high_usage_tasks']} tasks")
        report.append(f"    Very High (31+ calls): {usage_stats['very_high_usage_tasks']} tasks")

        # LLM calls by level
        if llm_stats['calls_by_level']:
            report.append("  LLM Calls by GAIA Level:")
            for level_name, level_stats in sorted(llm_stats['calls_by_level'].items(),
                                                 key=lambda x: int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 999):
                level_num = level_name.replace('level_', '')
                report.append(f"    Level {level_num}: Avg={level_stats['avg_calls']:.1f}, "
                            f"Max={level_stats['max_calls']}, Total={level_stats['total_calls']}")
        report.append("")
        
        report.append("🔍 Search & Planning Analysis:")
        report.append(f"  Average Search Actions: {search_stats['avg_searches']:.1f}")
        report.append(f"  Tasks with Search: {search_stats['tasks_with_search']}")
        report.append(f"  Tasks with Planning: {reasoning_stats['tasks_with_planning']}")
        report.append(f"  Tasks with Facts: {reasoning_stats['tasks_with_facts']}")
        report.append(f"  Tasks with Detailed Plan: {reasoning_stats['tasks_with_detailed_plan']}")
        report.append(f"  Average Facts Length: {reasoning_stats['avg_facts_words']:.0f} words")
        report.append(f"  Average Plan Length: {reasoning_stats['avg_plan_words_detailed']:.0f} words")
        report.append("")
        
        report.append("🛠️ Tool Usage Analysis:")
        report.append(f"  Average Tool Calls per Task: {tool_stats['avg_tool_calls_per_task']:.1f}")
        report.append(f"  Python Code Executions: {tool_stats['python_executions']}")
        report.append(f"  Search Agent Calls (detected): {tool_stats['search_agent_calls']}")
        report.append(f"  File Inspections (detected): {tool_stats['file_inspections']}")
        if tool_stats['most_used_tools']:
            report.append("  Most Used Tools:")
            for tool, count in list(tool_stats['most_used_tools'].items())[:5]:
                report.append(f"    - {tool}: {count}")
        report.append("")
        
        # Python Code Patterns Analysis
        if code_patterns:
            report.append("🐍 PYTHON CODE ANALYSIS:")
            report.append(f"  Total Search Agent Calls: {code_patterns['search_agent_usage']}")
            report.append(f"  Total File Inspections: {code_patterns['inspect_file_usage']}")
            report.append(f"  Average Code Complexity: {np.mean(code_patterns['code_complexity']):.1f} lines" if code_patterns['code_complexity'] else "  No code complexity data")
            
            if code_patterns['common_functions']:
                report.append("  Most Used Functions:")
                for func, count in list(code_patterns['common_functions'].most_common(10)):
                    if func not in ['print', 'len', 'str', 'int', 'float']:  # Skip basic Python functions
                        report.append(f"    - {func}(): {count}")
            
            if code_patterns['common_imports']:
                report.append("  Most Common Imports:")
                for imp, count in list(code_patterns['common_imports'].most_common(10)):
                    report.append(f"    - {imp}: {count}")
            report.append("")
        
        report.append("⚡ Action Success Analysis:")
        report.append(f"  Successful Actions: {action_stats['successful_actions']}")
        report.append(f"  Failed Actions: {action_stats['failed_actions']}")
        report.append(f"  Action Success Rate: {action_stats['success_rate']:.2%}")
        if action_stats['avg_step_duration'] > 0:
            report.append(f"  Average Step Duration: {action_stats['avg_step_duration']:.1f}s")
        report.append("")
        
        # Timing Analysis
        if 'error' not in timing_analysis:
            report.append("⏱️ TIMING ANALYSIS")
            report.append("-" * 40)
            report.append(f"Average Duration: {timing_analysis['avg_duration_seconds']:.1f}s")
            report.append(f"Median Duration: {timing_analysis['median_duration_seconds']:.1f}s")
            report.append(f"Max Duration: {timing_analysis['max_duration_seconds']:.1f}s")
            report.append(f"Min Duration: {timing_analysis['min_duration_seconds']:.1f}s")
            report.append(f"Total Runtime: {timing_analysis['total_runtime_seconds']:.1f}s ({timing_analysis['total_runtime_seconds']/3600:.1f}h)")

            # Level-specific duration analysis
            if timing_analysis.get('duration_analysis_by_level'):
                report.append("")
                report.append("📊 Duration Analysis by GAIA Level:")
                duration_by_level = timing_analysis['duration_analysis_by_level']

                for level_name, level_stats in sorted(duration_by_level.items(),
                                                    key=lambda x: int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 999):
                    level_num = level_name.replace('level_', '')
                    report.append(f"  Level {level_num}:")
                    report.append(f"    Average Duration: {level_stats['avg_duration']:.1f}s")
                    report.append(f"    Median Duration: {level_stats['median_duration']:.1f}s")
                    report.append(f"    Max Duration: {level_stats['max_duration']:.1f}s")
                    report.append(f"    Tasks: {level_stats['total_tasks']}")

                    # Special analysis for Level 3 (why it takes longer)
                    if level_num == '3' and 'duration_steps_correlation' in level_stats:
                        report.append("")
                        report.append("🔍 Level 3 Deep Analysis (Why Level 3 Takes Longer):")
                        report.append(f"    Average Steps: {level_stats['avg_steps']:.1f}")
                        report.append(f"    Average LLM Calls: {level_stats['avg_llm_calls']:.1f}")
                        report.append(f"    Average Search Actions: {level_stats['avg_search_actions']:.1f}")
                        report.append(f"    Error Rate: {level_stats['error_rate']:.2%}")
                        report.append(f"    Iteration Exceeded Rate: {level_stats['iteration_exceeded_rate']:.2%}")

                        report.append("    Correlations with Duration:")
                        report.append(f"      Duration vs Steps: {level_stats['duration_steps_correlation']:.3f}")
                        report.append(f"      Duration vs LLM Calls: {level_stats['duration_llm_correlation']:.3f}")
                        report.append(f"      Duration vs Search: {level_stats['duration_search_correlation']:.3f}")

                        if 'longest_tasks_characteristics' in level_stats:
                            longest_chars = level_stats['longest_tasks_characteristics']
                            report.append("    Characteristics of Longest Level 3 Tasks:")
                            report.append(f"      Average Duration: {longest_chars['avg_duration']:.1f}s")
                            report.append(f"      Average Steps: {longest_chars['avg_steps']:.1f}")
                            report.append(f"      Average LLM Calls: {longest_chars['avg_llm_calls']:.1f}")
                            report.append(f"      Average Search Actions: {longest_chars['avg_search_actions']:.1f}")
                            report.append(f"      Error Rate: {longest_chars['error_rate']:.2%}")
                            report.append(f"      Iteration Exceeded Rate: {longest_chars['iteration_exceeded_rate']:.2%}")
                            report.append(f"      Average Question Length: {longest_chars['avg_question_length']:.0f} chars")
                            report.append(f"      File Usage Rate: {longest_chars['files_usage_rate']:.2%}")

                            # Explanation of why Level 3 takes longer
                            report.append("")
                            report.append("💡 Analysis Summary - Why Level 3 Takes Longer:")
                            correlations = [
                                ("steps", level_stats['duration_steps_correlation']),
                                ("LLM calls", level_stats['duration_llm_correlation']),
                                ("search actions", level_stats['duration_search_correlation'])
                            ]
                            strongest_correlation = max(correlations, key=lambda x: abs(x[1]))

                            report.append(f"    Strongest correlation: Duration vs {strongest_correlation[0]} ({strongest_correlation[1]:.3f})")

                            if level_stats['avg_steps'] > 8:  # Assuming overall average is around 6-8
                                report.append("    Level 3 tasks require significantly more reasoning steps")
                            if level_stats['avg_llm_calls'] > 10:  # Threshold for high LLM usage
                                report.append("    Level 3 tasks involve more complex LLM interactions")
                            if level_stats['error_rate'] > 0.1:  # 10% error rate threshold
                                report.append("    Higher error rates contribute to longer execution times")
                            if level_stats['iteration_exceeded_rate'] > 0.1:
                                report.append("    Frequent iteration limit exceeded indicates complex problem-solving")

                    report.append("")

            report.append("")
        
        # Detailed Step Pattern Analysis
        if step_patterns:
            report.append("🔄 DETAILED STEP PATTERN ANALYSIS")
            report.append("-" * 40)
            
            # Common sequences
            report.append("Most Common Step Sequences:")
            for seq, count in list(step_patterns['common_sequences'].most_common(10)):
                report.append(f"  {seq}: {count} tasks")
            report.append("")
            
            # Step transitions
            report.append("Most Common Step Transitions:")
            for transition, count in list(step_patterns['step_type_transitions'].most_common(10)):
                report.append(f"  {transition}: {count} times")
            report.append("")
            
            # Success patterns
            flow_patterns = step_patterns['task_flow_patterns']
            if flow_patterns.get('success_rates_by_sequence'):
                report.append("Step Sequence Success Analysis:")
                # Sort by success rate
                sorted_sequences = sorted(
                    flow_patterns['success_rates_by_sequence'].items(),
                    key=lambda x: (x[1]['success_rate'], x[1]['total_count']),
                    reverse=True
                )
                
                for seq, stats in sorted_sequences[:10]:
                    if stats['total_count'] >= 2:  # Only show sequences with multiple instances
                        report.append(f"  {seq}")
                        report.append(f"    Success Rate: {stats['success_rate']:.1%} ({stats['success_count']}/{stats['total_count']})")
            report.append("")
        
        # Consistency Analysis (for multiple runs)
        if consistency_analysis and 'error' not in consistency_analysis:
            report.append("🔄 CONSISTENCY ANALYSIS ACROSS RUNS")
            report.append("-" * 40)
            
            pred_consistency = consistency_analysis['prediction_consistency']
            report.append(f"Total Runs Analyzed: {consistency_analysis['total_runs']}")
            report.append(f"Common Tasks: {consistency_analysis['common_tasks']}")
            report.append(f"Identical Predictions: {pred_consistency['identical_predictions']}/{consistency_analysis['common_tasks']} ({pred_consistency['identical_predictions_rate']:.1%})")
            report.append(f"Consistent Accuracy: {pred_consistency['consistent_accuracy']}/{consistency_analysis['common_tasks']} ({pred_consistency['consistent_accuracy_rate']:.1%})")
            report.append("")
            
            # Run-by-run comparison
            if consistency_analysis.get('run_comparison'):
                report.append("Run-by-Run Agreement Rates:")
                for comparison, stats in consistency_analysis['run_comparison'].items():
                    report.append(f"  {comparison}: {stats['agreement_rate']:.1%} ({stats['agreement_count']}/{stats['total_compared']})")
                report.append("")
            
            # Variance analysis
            acc_var = consistency_analysis['accuracy_variance']
            duration_var = consistency_analysis['duration_variance'] 
            step_var = consistency_analysis['step_count_variance']
            
            report.append("Variance Metrics:")
            report.append(f"  Accuracy Variance: Mean={acc_var['mean_variance']:.3f}, Max={acc_var['max_variance']:.3f}")
            if duration_var['mean_variance'] > 0:
                report.append(f"  Duration Variance: Mean={duration_var['mean_variance']:.1f}s², Max={duration_var['max_variance']:.1f}s²")
            report.append(f"  Step Count Variance: Mean={step_var['mean_variance']:.2f}, Max={step_var['max_variance']:.2f}")
            report.append("")
            
            # Most inconsistent tasks
            if consistency_analysis.get('most_inconsistent_tasks'):
                report.append("Most Inconsistent Tasks (Different Predictions):")
                for task_info in consistency_analysis['most_inconsistent_tasks'][:5]:
                    report.append(f"  Task ID: {task_info['task_id'][:8]}...")
                    report.append(f"    Unique Predictions: {task_info['unique_predictions']}")
                    report.append(f"    Predictions: {task_info['predictions'][:3]}...")  # Show first 3
                report.append("")
            
            # Most consistent tasks  
            consistent_tasks_count = sum(1 for t in consistency_analysis.get('most_consistent_tasks', []) if t['all_same'])
            report.append(f"Fully Consistent Tasks: {consistent_tasks_count}/{consistency_analysis['common_tasks']}")
            report.append("")
        
        # Top Failed Tasks
        report.append("🔍 FAILED TASKS ANALYSIS")
        report.append("-" * 40)
        failed_tasks = [r for r in self.results 
                       if r.get('prediction', '').lower().strip() != r.get('true_answer', '').lower().strip()]
        
        if failed_tasks:
            report.append(f"Failed Tasks: {len(failed_tasks)}")
            
            # Show first few failed tasks for investigation
            report.append("\nSample Failed Tasks:")
            for i, task in enumerate(failed_tasks[:5]):
                report.append(f"{i+1}. Task ID: {task.get('task_id', 'N/A')}")
                report.append(f"   Question: {task.get('question', 'N/A')[:100]}...")
                report.append(f"   Predicted: {task.get('prediction', 'N/A')}")
                report.append(f"   Expected: {task.get('true_answer', 'N/A')}")
                if task.get('agent_error'):
                    report.append(f"   Error: {task.get('agent_error', 'N/A')[:100]}...")
                report.append("")
        
        return "\n".join(report)
    
    def export_detailed_csv(self, output_file: str = "gaia_detailed_results.csv") -> None:
        """Export detailed results to CSV for further analysis"""
        if self.df is None or self.df.empty:
            print("No data to export")
            return
        
        # Flatten some nested data for CSV export
        export_df = self.df.copy()
        
        # Add computed columns
        export_df['is_correct'] = (export_df['prediction'].str.lower().str.strip() == 
                                  export_df['true_answer'].str.lower().str.strip())
        
        export_df['has_agent_error'] = export_df['agent_error'].notna()
        export_df['has_parsing_error'] = export_df['parsing_error'].notna()
        
        # Calculate duration if possible
        durations = []
        for _, row in export_df.iterrows():
            start_time = row.get('start_time')
            end_time = row.get('end_time')
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    durations.append(duration)
                except:
                    durations.append(None)
            else:
                durations.append(None)
        
        export_df['duration_seconds'] = durations
        export_df['num_intermediate_steps'] = export_df['intermediate_steps'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        export_df['num_search_actions'] = export_df['search_agent_actions'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Add LLM call count estimation
        def estimate_llm_calls(row):
            steps = row.get('intermediate_steps', [])
            if not isinstance(steps, list):
                return 0

            planning_steps = [s for s in steps if isinstance(s, dict) and s.get('step_type') == 'planning']
            action_steps = [s for s in steps if isinstance(s, dict) and s.get('step_type') == 'action']
            search_records = row.get('search_agent_actions', [])
            search_count = len(search_records) if isinstance(search_records, list) else 0

            return len(planning_steps) + len(action_steps) + search_count

        export_df['estimated_llm_calls'] = export_df.apply(estimate_llm_calls, axis=1)
        
        # Select columns for export
        columns_to_export = [
            'source_file', 'agent_name', 'task_id', 'task', 'question',
            'prediction', 'true_answer', 'is_correct',
            'has_agent_error', 'has_parsing_error', 'iteration_limit_exceeded',
            'duration_seconds', 'num_intermediate_steps', 'num_search_actions', 'estimated_llm_calls',
            'start_time', 'end_time'
        ]
        
        columns_to_export = [col for col in columns_to_export if col in export_df.columns]
        
        export_df[columns_to_export].to_csv(output_file, index=False)
        print(f"Detailed results exported to {output_file}")
    
    def _calculate_avg_duration(self, results_subset) -> float:
        """Calculate average duration for a subset of results"""
        durations = []
        for _, result in results_subset.iterrows():
            start_time = result.get('start_time')
            end_time = result.get('end_time')
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    durations.append(duration)
                except:
                    continue
        return np.mean(durations) if durations else 0

    def _analyze_llm_calls_by_level(self, calls_by_task: List[Dict]) -> Dict[str, Any]:
        """Analyze LLM call patterns by GAIA level"""
        level_stats = defaultdict(list)

        for task_info in calls_by_task:
            level = task_info['level']
            llm_calls = task_info['llm_calls']
            level_stats[level].append(llm_calls)

        result = {}
        for level, calls_list in level_stats.items():
            result[f'level_{level}'] = {
                'avg_calls': np.mean(calls_list),
                'median_calls': np.median(calls_list),
                'max_calls': max(calls_list),
                'min_calls': min(calls_list),
                'total_tasks': len(calls_list),
                'total_calls': sum(calls_list)
            }

        return result

    def _categorize_tasks_by_llm_usage(self, llm_calls_per_task: List[int]) -> Dict[str, int]:
        """Categorize tasks by LLM usage intensity"""
        if not llm_calls_per_task:
            return {}

        low_usage = sum(1 for x in llm_calls_per_task if x <= 5)
        medium_usage = sum(1 for x in llm_calls_per_task if 6 <= x <= 15)
        high_usage = sum(1 for x in llm_calls_per_task if 16 <= x <= 30)
        very_high_usage = sum(1 for x in llm_calls_per_task if x > 30)

        return {
            'low_usage_tasks': low_usage,      # 1-5 calls
            'medium_usage_tasks': medium_usage, # 6-15 calls
            'high_usage_tasks': high_usage,     # 16-30 calls
            'very_high_usage_tasks': very_high_usage # 31+ calls
        }

    def _analyze_duration_by_level(self) -> Dict[str, Any]:
        """Analyze duration patterns by GAIA level to understand why level 3 takes longer"""
        if not self.results:
            return {}

        level_analysis = defaultdict(lambda: {
            'durations': [],
            'steps': [],
            'llm_calls': [],
            'search_actions': [],
            'errors': 0,
            'iteration_exceeded': 0,
            'task_details': []
        })

        for result in self.results:
            level = result.get('task', 'unknown')
            start_time = result.get('start_time')
            end_time = result.get('end_time')

            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    level_analysis[level]['durations'].append(duration)
                except:
                    continue

            # Collect other metrics
            steps = result.get('intermediate_steps', [])
            search_records = result.get('search_agent_actions', [])

            level_analysis[level]['steps'].append(len(steps))
            level_analysis[level]['search_actions'].append(len(search_records))

            # Count LLM calls (estimated)
            planning_steps = [s for s in steps if s.get('step_type') == 'planning']
            action_steps = [s for s in steps if s.get('step_type') == 'action']
            estimated_llm_calls = len(planning_steps) + len(action_steps) + len(search_records)
            level_analysis[level]['llm_calls'].append(estimated_llm_calls)

            # Count errors
            if result.get('agent_error') or result.get('parsing_error'):
                level_analysis[level]['errors'] += 1

            if result.get('iteration_limit_exceeded'):
                level_analysis[level]['iteration_exceeded'] += 1

            # Store task details for level 3 analysis
            if level == 3:
                level_analysis[level]['task_details'].append({
                    'task_id': result.get('task_id', 'unknown'),
                    'duration': duration if 'duration' in locals() else None,
                    'steps': len(steps),
                    'search_actions': len(search_records),
                    'llm_calls': estimated_llm_calls,
                    'has_error': bool(result.get('agent_error') or result.get('parsing_error')),
                    'iteration_exceeded': result.get('iteration_limit_exceeded', False),
                    'question_length': len(result.get('question', '')),
                    'has_files': bool(result.get('file_name')),
                    'prediction_length': len(result.get('prediction', ''))
                })

        # Calculate statistics for each level
        result = {}
        for level, data in level_analysis.items():
            if data['durations']:
                duration_stats = {
                    'avg_duration': np.mean(data['durations']),
                    'median_duration': np.median(data['durations']),
                    'max_duration': max(data['durations']),
                    'min_duration': min(data['durations']),
                    'std_duration': np.std(data['durations']),
                    'total_tasks': len(data['durations'])
                }

                # Calculate correlations for level 3
                if level == 3 and len(data['durations']) > 1:
                    try:
                        # Correlation between duration and steps
                        duration_steps_corr = np.corrcoef(data['durations'], data['steps'])[0, 1]
                        # Correlation between duration and LLM calls
                        duration_llm_corr = np.corrcoef(data['durations'], data['llm_calls'])[0, 1]
                        # Correlation between duration and search actions
                        duration_search_corr = np.corrcoef(data['durations'], data['search_actions'])[0, 1]

                        duration_stats.update({
                            'duration_steps_correlation': duration_steps_corr,
                            'duration_llm_correlation': duration_llm_corr,
                            'duration_search_correlation': duration_search_corr,
                            'avg_steps': np.mean(data['steps']),
                            'avg_llm_calls': np.mean(data['llm_calls']),
                            'avg_search_actions': np.mean(data['search_actions']),
                            'error_rate': data['errors'] / len(data['durations']),
                            'iteration_exceeded_rate': data['iteration_exceeded'] / len(data['durations'])
                        })

                        # Analyze top longest tasks for level 3
                        if data['task_details']:
                            sorted_tasks = sorted(data['task_details'],
                                                key=lambda x: x['duration'] or 0, reverse=True)
                            duration_stats['longest_tasks_analysis'] = sorted_tasks[:5]

                            # Analyze characteristics of longest tasks
                            longest_tasks = sorted_tasks[:3]  # Top 3 longest
                            avg_longest = {
                                'avg_duration': np.mean([t['duration'] for t in longest_tasks if t['duration']]),
                                'avg_steps': np.mean([t['steps'] for t in longest_tasks]),
                                'avg_llm_calls': np.mean([t['llm_calls'] for t in longest_tasks]),
                                'avg_search_actions': np.mean([t['search_actions'] for t in longest_tasks]),
                                'error_rate': sum(t['has_error'] for t in longest_tasks) / len(longest_tasks),
                                'iteration_exceeded_rate': sum(t['iteration_exceeded'] for t in longest_tasks) / len(longest_tasks),
                                'avg_question_length': np.mean([t['question_length'] for t in longest_tasks]),
                                'files_usage_rate': sum(t['has_files'] for t in longest_tasks) / len(longest_tasks)
                            }
                            duration_stats['longest_tasks_characteristics'] = avg_longest

                    except (ValueError, np.linalg.LinAlgError):
                        # Handle cases where correlation cannot be computed
                        pass

                result[f'level_{level}'] = duration_stats

        return result
    
    def analyze_detailed_step_patterns(self) -> Dict[str, Any]:
        """Analyze detailed patterns in intermediate steps"""
        if not self.results:
            return {}
        
        step_patterns = {
            'task_flow_patterns': {},
            'common_sequences': Counter(),
            'error_patterns': [],
            'successful_patterns': [],
            'step_type_transitions': Counter()
        }
        
        for result in self.results:
            steps = result.get('intermediate_steps', [])
            is_correct = (result.get('prediction', '').lower().strip() == 
                         result.get('true_answer', '').lower().strip())
            
            # Analyze step sequence
            step_types = [step.get('step_type', 'unknown') for step in steps]
            sequence = ' -> '.join(step_types)
            step_patterns['common_sequences'][sequence] += 1
            
            # Analyze transitions between step types
            for i in range(len(step_types) - 1):
                transition = f"{step_types[i]} -> {step_types[i+1]}"
                step_patterns['step_type_transitions'][transition] += 1
            
            # Categorize patterns by success
            pattern_info = {
                'sequence': sequence,
                'length': len(steps),
                'task_id': result.get('task_id', 'unknown'),
                'has_errors': bool(result.get('agent_error') or result.get('parsing_error')),
                'prediction': result.get('prediction', ''),
                'true_answer': result.get('true_answer', '')
            }
            
            if is_correct:
                step_patterns['successful_patterns'].append(pattern_info)
            else:
                step_patterns['error_patterns'].append(pattern_info)
        
        # Analyze most effective patterns
        successful_sequences = Counter()
        failed_sequences = Counter()
        
        for pattern in step_patterns['successful_patterns']:
            successful_sequences[pattern['sequence']] += 1
            
        for pattern in step_patterns['error_patterns']:
            failed_sequences[pattern['sequence']] += 1
        
        step_patterns['task_flow_patterns'] = {
            'most_successful_sequences': dict(successful_sequences.most_common(5)),
            'most_failed_sequences': dict(failed_sequences.most_common(5)),
            'success_rates_by_sequence': {}
        }
        
        # Calculate success rates for each sequence
        all_sequences = set(successful_sequences.keys()) | set(failed_sequences.keys())
        for seq in all_sequences:
            success_count = successful_sequences.get(seq, 0)
            fail_count = failed_sequences.get(seq, 0)
            total = success_count + fail_count
            success_rate = success_count / total if total > 0 else 0
            step_patterns['task_flow_patterns']['success_rates_by_sequence'][seq] = {
                'success_rate': success_rate,
                'total_count': total,
                'success_count': success_count,
                'fail_count': fail_count
            }
        
        return step_patterns
    
    def analyze_python_code_patterns(self) -> Dict[str, Any]:
        """Analyze patterns within Python code execution"""
        if not self.results:
            return {}
        
        code_patterns = {
            'search_agent_usage': 0,
            'inspect_file_usage': 0,
            'common_imports': Counter(),
            'common_functions': Counter(),
            'code_complexity': []
        }
        
        for result in self.results:
            steps = result.get('intermediate_steps', [])
            
            for step in steps:
                if step.get('step_type') == 'action':
                    tool_calls = step.get('tool_calls', [])
                    
                    for tool_call in tool_calls:
                        if (isinstance(tool_call, dict) and 
                            tool_call.get('function', {}).get('name') == 'python_interpreter'):
                            
                            code = tool_call.get('function', {}).get('arguments', '')
                            
                            # Count specific function calls
                            search_count = code.count('search_agent(')
                            inspect_count = code.count('inspect_file')
                            code_patterns['search_agent_usage'] += search_count
                            code_patterns['inspect_file_usage'] += inspect_count
                            
                            # Analyze imports
                            import_lines = [line.strip() for line in code.split('\n') 
                                          if line.strip().startswith(('import ', 'from '))]
                            for import_line in import_lines:
                                code_patterns['common_imports'][import_line] += 1
                            
                            # Analyze function calls
                            import re
                            function_calls = re.findall(r'(\w+)\s*\(', code)
                            for func in function_calls:
                                code_patterns['common_functions'][func] += 1
                            
                            # Simple complexity metric
                            lines = code.split('\n')
                            non_empty_lines = [l for l in lines if l.strip()]
                            code_patterns['code_complexity'].append(len(non_empty_lines))
        
        return code_patterns
    
    def analyze_consistency_across_runs(self) -> Dict[str, Any]:
        """Analyze consistency of results across multiple runs of the same tasks"""
        if not self.results:
            return {}
        
        # Group results by source file and task_id
        runs_by_file = defaultdict(dict)
        for result in self.results:
            source_file = result.get('source_file', 'unknown')
            task_id = result.get('task_id', 'unknown')
            runs_by_file[source_file][task_id] = result
        
        if len(runs_by_file) < 2:
            return {"error": "Need at least 2 runs for consistency analysis"}
        
        # Find common tasks across all runs
        all_task_ids = set()
        for file_results in runs_by_file.values():
            all_task_ids.update(file_results.keys())
        
        common_tasks = all_task_ids
        for file_results in runs_by_file.values():
            common_tasks = common_tasks.intersection(set(file_results.keys()))
        
        if not common_tasks:
            return {"error": "No common tasks found across runs"}
        
        consistency_metrics = {
            'total_runs': len(runs_by_file),
            'common_tasks': len(common_tasks),
            'prediction_consistency': {},
            'accuracy_variance': {},
            'duration_variance': {},
            'step_count_variance': {},
            'most_inconsistent_tasks': [],
            'most_consistent_tasks': [],
            'run_comparison': {}
        }
        
        # Analyze prediction consistency for each task
        task_consistency = {}
        for task_id in common_tasks:
            task_results = []
            predictions = []
            accuracies = []
            durations = []
            step_counts = []
            
            for file_name, file_results in runs_by_file.items():
                if task_id in file_results:
                    result = file_results[task_id]
                    task_results.append((file_name, result))
                    
                    prediction = result.get('prediction', '').lower().strip()
                    true_answer = result.get('true_answer', '').lower().strip()
                    predictions.append(prediction)
                    accuracies.append(1 if prediction == true_answer else 0)
                    
                    # Duration
                    start_time = result.get('start_time')
                    end_time = result.get('end_time') 
                    if start_time and end_time:
                        try:
                            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                            duration = (end_dt - start_dt).total_seconds()
                            durations.append(duration)
                        except:
                            durations.append(None)
                    else:
                        durations.append(None)
                    
                    # Step count
                    steps = result.get('intermediate_steps', [])
                    step_counts.append(len(steps))
            
            # Calculate consistency metrics for this task
            unique_predictions = len(set(predictions))
            all_correct = all(acc == 1 for acc in accuracies)
            all_wrong = all(acc == 0 for acc in accuracies)
            accuracy_variance = np.var(accuracies) if len(accuracies) > 1 else 0
            
            valid_durations = [d for d in durations if d is not None]
            duration_variance = np.var(valid_durations) if len(valid_durations) > 1 else 0
            step_variance = np.var(step_counts) if len(step_counts) > 1 else 0
            
            task_consistency[task_id] = {
                'unique_predictions': unique_predictions,
                'all_same_prediction': unique_predictions == 1,
                'all_correct': all_correct,
                'all_wrong': all_wrong,
                'accuracy_variance': accuracy_variance,
                'duration_variance': duration_variance,
                'step_count_variance': step_variance,
                'predictions': predictions,
                'accuracies': accuracies,
                'durations': valid_durations,
                'step_counts': step_counts,
                'consistency_score': 1.0 if unique_predictions == 1 else 0.0
            }
        
        # Overall consistency metrics
        total_consistent_predictions = sum(1 for t in task_consistency.values() if t['all_same_prediction'])
        total_consistent_accuracy = sum(1 for t in task_consistency.values() if t['all_correct'] or t['all_wrong'])
        
        consistency_metrics['prediction_consistency'] = {
            'identical_predictions': total_consistent_predictions,
            'identical_predictions_rate': total_consistent_predictions / len(common_tasks),
            'consistent_accuracy': total_consistent_accuracy,
            'consistent_accuracy_rate': total_consistent_accuracy / len(common_tasks)
        }
        
        # Variance metrics
        all_accuracy_vars = [t['accuracy_variance'] for t in task_consistency.values()]
        all_duration_vars = [t['duration_variance'] for t in task_consistency.values() if t['duration_variance'] > 0]
        all_step_vars = [t['step_count_variance'] for t in task_consistency.values()]
        
        consistency_metrics['accuracy_variance'] = {
            'mean_variance': np.mean(all_accuracy_vars),
            'max_variance': max(all_accuracy_vars) if all_accuracy_vars else 0
        }
        
        consistency_metrics['duration_variance'] = {
            'mean_variance': np.mean(all_duration_vars) if all_duration_vars else 0,
            'max_variance': max(all_duration_vars) if all_duration_vars else 0
        }
        
        consistency_metrics['step_count_variance'] = {
            'mean_variance': np.mean(all_step_vars),
            'max_variance': max(all_step_vars) if all_step_vars else 0
        }
        
        # Most/least consistent tasks
        sorted_by_consistency = sorted(task_consistency.items(), 
                                     key=lambda x: x[1]['consistency_score'], reverse=True)
        
        consistency_metrics['most_consistent_tasks'] = [
            {
                'task_id': task_id,
                'consistency_score': data['consistency_score'],
                'predictions': data['predictions'],
                'all_same': data['all_same_prediction']
            }
            for task_id, data in sorted_by_consistency[:10]
        ]
        
        consistency_metrics['most_inconsistent_tasks'] = [
            {
                'task_id': task_id,
                'consistency_score': data['consistency_score'],
                'predictions': data['predictions'],
                'unique_predictions': data['unique_predictions']
            }
            for task_id, data in sorted_by_consistency[-10:] if data['consistency_score'] < 1.0
        ]
        
        # Run-by-run comparison
        file_names = list(runs_by_file.keys())
        for i, file1 in enumerate(file_names):
            for file2 in file_names[i+1:]:
                agreement_count = 0
                total_compared = 0
                
                for task_id in common_tasks:
                    if task_id in runs_by_file[file1] and task_id in runs_by_file[file2]:
                        pred1 = runs_by_file[file1][task_id].get('prediction', '').lower().strip()
                        pred2 = runs_by_file[file2][task_id].get('prediction', '').lower().strip()
                        if pred1 == pred2:
                            agreement_count += 1
                        total_compared += 1
                
                comparison_key = f"{file1} vs {file2}"
                consistency_metrics['run_comparison'][comparison_key] = {
                    'agreement_count': agreement_count,
                    'total_compared': total_compared,
                    'agreement_rate': agreement_count / total_compared if total_compared > 0 else 0
                }
        
        return consistency_metrics
    
    def create_visualizations(self, output_dir: str = "./analysis_plots") -> None:
        """Create visualization plots for the analysis"""
        if self.df is None or self.df.empty:
            print("No data available for visualization")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 0. Task Level Distribution
        level_distribution = self.analyze_task_level_distribution()
        if level_distribution and level_distribution['level_distribution']:
            plt.figure(figsize=(10, 6))
            
            # Prepare data for plotting, excluding unknown levels if there are known levels
            valid_distributions = {k: v for k, v in level_distribution['level_distribution'].items() 
                                 if not k.endswith('_unknown') or len(level_distribution['level_distribution']) == 1}
            
            if valid_distributions:
                levels = []
                counts = []
                
                # Sort by level number for proper display
                sorted_items = sorted(valid_distributions.items(), 
                                    key=lambda x: int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 999)
                
                for level_name, stats in sorted_items:
                    level_num = level_name.replace('level_', '')
                    if level_num == 'unknown':
                        levels.append("Unknown Level")
                    else:
                        levels.append(f"Level {level_num}")
                    counts.append(stats['count'])
                
                # Create bar plot
                bars = plt.bar(levels, counts, alpha=0.7, edgecolor='black')
                plt.title('GAIA Problem Level Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('GAIA Level')
                plt.ylabel('Number of Tasks')
                
                # Add percentage labels on bars
                total = sum(counts)
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{count}\n({count/total*100:.1f}%)',
                            ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_path / 'gaia_level_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 1. Accuracy by Task Level
        if 'gaia_level' in self.df.columns and (self.df['gaia_level'] != 'unknown').any():
            plt.figure(figsize=(10, 6))
            
            # Filter out unknown levels for accuracy analysis
            valid_levels_df = self.df[self.df['gaia_level'] != 'unknown']
            
            if not valid_levels_df.empty:
                task_level_accuracy = valid_levels_df.groupby('gaia_level').apply(
                    lambda x: (x['prediction'].str.lower().str.strip() == 
                              x['true_answer'].str.lower().str.strip()).mean()
                )
                
                # Sort by level number
                try:
                    task_level_accuracy = task_level_accuracy.reindex(
                        sorted(task_level_accuracy.index, key=lambda x: int(x) if str(x).isdigit() else 999)
                    )
                except:
                    pass
                
                # Create labels with "Level" prefix
                labels = [f"Level {level}" for level in task_level_accuracy.index]
                
                plt.bar(labels, task_level_accuracy.values, alpha=0.7, edgecolor='black')
                plt.title('Accuracy by GAIA Task Level', fontsize=14, fontweight='bold')
                plt.xlabel('GAIA Level')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45)
                
                # Add accuracy values on bars
                for i, acc in enumerate(task_level_accuracy.values):
                    plt.text(i, acc + 0.01, f'{acc:.2%}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_path / 'accuracy_by_task_level.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Error Distribution
        plt.figure(figsize=(12, 6))
        error_counts = {
            'Agent Errors': self.df['agent_error'].notna().sum(),
            'Parsing Errors': self.df['parsing_error'].notna().sum(),
            'Iteration Exceeded': self.df['iteration_limit_exceeded'].sum(),
            'Successful': (~self.df['agent_error'].notna() & ~self.df['parsing_error'].notna() & 
                          ~self.df['iteration_limit_exceeded']).sum()
        }
        
        plt.bar(error_counts.keys(), error_counts.values())
        plt.title('Distribution of Task Outcomes')
        plt.ylabel('Number of Tasks')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Steps vs Performance
        if 'num_intermediate_steps' not in self.df.columns:
            self.df['num_intermediate_steps'] = self.df['intermediate_steps'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        plt.figure(figsize=(10, 6))
        self.df['is_correct'] = (self.df['prediction'].str.lower().str.strip() == 
                                self.df['true_answer'].str.lower().str.strip())
        
        correct_steps = self.df[self.df['is_correct']]['num_intermediate_steps']
        incorrect_steps = self.df[~self.df['is_correct']]['num_intermediate_steps']
        
        plt.hist([correct_steps, incorrect_steps], bins=20, alpha=0.7, 
                 label=['Correct', 'Incorrect'], edgecolor='black')
        plt.xlabel('Number of Intermediate Steps')
        plt.ylabel('Frequency')
        plt.title('Distribution of Steps: Correct vs Incorrect Answers')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'steps_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze GAIA benchmark results')
    parser.add_argument('--output_dir', type=str, default='./output/validation',
                       help='Directory containing JSONL result files')
    parser.add_argument('--pattern', type=str, default='*.jsonl',
                       help='File pattern to match (default: *.jsonl)')
    parser.add_argument('--export_csv', action='store_true',
                       help='Export detailed results to CSV')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--plot_dir', type=str, default='./analysis_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GAIAResultsAnalyzer(args.output_dir)
    
    # Load results
    try:
        analyzer.load_results(args.pattern)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Generate report
    print(analyzer.generate_detailed_report())
    
    # Export CSV if requested
    if args.export_csv:
        analyzer.export_detailed_csv()
    
    # Create plots if requested
    if args.create_plots:
        analyzer.create_visualizations(args.plot_dir)


if __name__ == "__main__":
    main()