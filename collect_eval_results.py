#!/usr/bin/env python3
"""Collect `Rouge-L f` and `str_match` from eval_result.json files.

Usage:
  python collect_eval_results.py  # uses default SVBench path
  python collect_eval_results.py --base /path/to/SVBench

The script searches for files matching
    <base>/*/answers/*/eval_result.json
and writes two paper-style wide tables with methods on rows and tasks on columns:
one table for `Rouge-L f`, one table for `str_match`. Each table also includes
the mean over all QA metrics and the average of subtask metrics.
"""
import os
import json
import argparse
import csv
import glob
import sys


TASK_ORDER = [
    'action_classification',
    'commentary_generation',
    'fouls_penalties',
    'offside_judgement',
    'player_identification',
    'space_identification',
    'time_allocation',
]

METHOD_ORDER = ['fps_1.0', 'Uniform_32', 'ASK_32', 'FOCUS_32', 'KFC_32']


def find_eval_files(base_dir):
    pattern = os.path.join(base_dir, '*', 'answers', '*', 'eval_result.json')
    return glob.glob(pattern, recursive=True)


def to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_metrics(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {
            'rouge': None,
            'str_match': None,
            'rouge_values': [],
            'str_match_values': [],
        }

    rouge = to_float(data.get('Rouge-L f'))
    str_match = to_float(data.get('str_match'))
    rouge_values = []
    str_match_values = []
    for item in data.get('eval_list', []):
        item_rouge = to_float(item.get('Rouge-L'))
        if item_rouge is not None:
            rouge_values.append(item_rouge)

        item_str_match = to_float(item.get('str_match'))
        if item_str_match is not None:
            str_match_values.append(item_str_match)

    return {
        'rouge': rouge,
        'str_match': str_match,
        'rouge_values': rouge_values,
        'str_match_values': str_match_values,
    }


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def format_metric(value):
    return '-' if value is None else f'{value:.6g}'


def write_csv(headers, rows, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)


def print_table(headers, rows):
    # simple column width calc
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(value)))

    fmt = '  '.join('{:<%d}' % w for w in col_widths)
    print(fmt.format(*headers))
    print('-' * (sum(col_widths) + 4))
    for row in rows:
        print(fmt.format(*row))


def build_metric_tables(files, base_dir):
    records = {}
    methods = []
    tasks = []

    for fp in files:
        metrics = read_metrics(fp)
        rel = os.path.relpath(fp, base_dir)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue

        task = parts[0]
        method = parts[2]

        if method not in methods:
            methods.append(method)
        if task not in tasks:
            tasks.append(task)

        records.setdefault(method, {})[task] = metrics

    ordered_methods = [
        method for method in METHOD_ORDER if method in records
    ] + [
        method for method in methods if method not in METHOD_ORDER
    ]
    ordered_tasks = [
        task for task in TASK_ORDER if task in tasks
    ] + [
        task for task in tasks if task not in TASK_ORDER
    ]

    rouge_rows = []
    sm_rows = []
    for method in ordered_methods:
        rouge_row = [method]
        sm_row = [method]
        method_records = records.get(method, {})
        task_rouges = []
        task_str_matches = []
        benchmark_rouges = []
        benchmark_str_matches = []

        for task in ordered_tasks:
            value = method_records.get(task)
            if value is None:
                rouge_row.append('-')
                sm_row.append('-')
            else:
                rouge = value['rouge']
                sm = value['str_match']
                rouge_row.append(format_metric(rouge))
                sm_row.append(format_metric(sm))
                task_rouges.append(rouge)
                task_str_matches.append(sm)
                benchmark_rouges.extend(value['rouge_values'])
                benchmark_str_matches.extend(value['str_match_values'])

        # `Total` is the mean of all QA metrics over all available subtasks.
        # If eval_list is missing, fall back to the mean of task-level metrics.
        benchmark_rouge = mean(benchmark_rouges) if benchmark_rouges else mean(task_rouges)
        benchmark_str_match = mean(benchmark_str_matches) if benchmark_str_matches else mean(task_str_matches)
        # import ipdb; ipdb.set_trace()
        rouge_row.extend([format_metric(benchmark_rouge), format_metric(mean(task_rouges))])
        sm_row.extend([format_metric(benchmark_str_match), format_metric(mean(task_str_matches))])
        rouge_rows.append(rouge_row)
        sm_rows.append(sm_row)

    headers = ['method'] + ordered_tasks + ['Total', 'Avg.']
    return headers, rouge_rows, sm_rows


def main():
    default_base = os.path.join(
        'eval_output', 'qwen3_vl', 'SVBench'
    )

    p = argparse.ArgumentParser()
    p.add_argument('--base', default=default_base, help='SVBench base directory')
    p.add_argument('--out-rouge', default=None, help='Rouge-L f CSV output path')
    p.add_argument('--out-str', default=None, help='str_match CSV output path')
    p.add_argument('--no-print', action='store_true', help='Do not pretty-print the table')
    args = p.parse_args()

    base = args.base
    if not os.path.isdir(base):
        print('Base directory not found:', base, file=sys.stderr)
        sys.exit(2)

    if args.out_rouge is None:
        args.out_rouge = os.path.join(base, 'eval_summary_rouge.csv')
    if args.out_str is None:
        args.out_str = os.path.join(base, 'eval_summary_str_match.csv')

    files = find_eval_files(base)
    files.sort()
    headers, rouge_rows, sm_rows = build_metric_tables(files, base)

    write_csv(headers, rouge_rows, args.out_rouge)
    write_csv(headers, sm_rows, args.out_str)

    if not args.no_print:
        print('Rouge-L f table')
        print_table(headers, rouge_rows)
        print()
        print('str_match table')
        print_table(headers, sm_rows)

    print('\nWrote', args.out_rouge)
    print('Wrote', args.out_str)


if __name__ == '__main__':
    main()
