#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.ticker import AutoMinorLocator
from binary import BinaryUnits, DecimalUnits, convert_units
from pathlib import Path
import re
import datetime
import glob
import json
import argparse

colors = [
    '#72AB97',
    '#728CA6',
    '#FFDCAA',
    '#FFC6AA',
]

def indexes():
    return ['config','bs','qd','jobcount','workload']

def indexes_no_config():
    i = indexes()
    i.remove('config')
    return i

def get_columns(index):
    columns = set(indexes())
    return columns - set([index]) - set(['config'])

def load_log(path):
    glob_path = f'{path}/log-*.log'

    try:
        log_path = glob.glob(glob_path)[0]
    except IndexError as e:
        print(f"Could not find log file at {glob_path}")
        raise e

    with open(log_path, "rt") as f:
        log_data = f.read()

    return log_data

def get_ip_from_log(log_data):
    match = re.search(r"\d+\.\d+\.\d+\.\d+", log_data)
    if not match:
        return "No IP"
    ip = match.group(0)
    return ip

def get_lang_from_log(log_data):
    match = re.search(r"null_blk", log_data)
    if match:
        return "c"
    else:
        return "rust"

def load_file(path, config):
    files = glob.glob(f'{path}/**/*.json')
    frame = pd.DataFrame()

    log_data = load_log(f'{path}')
    ip = get_ip_from_log(log_data)
    lang = get_lang_from_log(log_data)

    for jfile in files:
        with open(jfile, "rt") as f:
            data = json.load(f)

        timestamp = datetime.datetime.fromtimestamp(data['timestamp'])

        job = data['jobs'][0]
        options = job['job options']

        if options['bs'].endswith('m'):
            bs = int(options['bs'].rstrip('m')) * 1024 * 1024
        elif options['bs'].endswith('k'):
            bs = int(options['bs'].rstrip('k')) * 1024
        else:
            bs = int(options['bs'])

        qd = int(options['iodepth'])
        iops = int(data["jobs"][0]["read"]["iops"]) + int(data["jobs"][0]["write"]["iops"])
        jobcount = int(options['numjobs'])
        workload = options['rw']

        new = pd.DataFrame({
            'config': config,
            'lang': lang,
            'qd': qd,
            'bs': bs,
            'jobcount': jobcount,
            'workload': workload,
            'iops': iops,
            'ip': ip,
            'timestamp': timestamp,
        }, index=[0])
        frame = pd.concat([frame, new], ignore_index=True)
    return frame

def append_single(frame, path, config):
    frame = pd.concat([frame, load_file(path, config)], ignore_index=True)
    return frame

def calculate_difference(frame, a, b):
    group = frame\
        .groupby(indexes())['iops']

    stat = pd.DataFrame({
        "samples": group.count(),
        "mean": group.mean(),
        "variance": group.var(),
        "stddev": group.std(),
    }).reset_index().pivot(index=indexes_no_config(), columns=['config']).sort_index(level=['qd'])

    confidence = 95
    tval = stat['samples'][a].map(lambda x: np.abs(sp.stats.t.ppf((100-confidence) / 200, x)))
    stderr = ( (stat['variance'][a] / stat['samples'][a]) + (stat['variance'][b] / stat['samples'][b]) ).apply(np.sqrt)
    interval = stderr * tval

    tval_a = stat['samples'][a].map(lambda x: np.abs(sp.stats.t.ppf((100-confidence) / 200, x)))
    stderr_a = stat['variance'][a] / stat['samples'][a].apply(np.sqrt)
    interval_a = stderr * tval

    tval_b = stat['samples'][b].map(lambda x: np.abs(sp.stats.t.ppf((100-confidence) / 200, x)))
    stderr_b = stat['variance'][b] / stat['samples'][b].apply(np.sqrt)
    interval_b = stderr * tval

    result = pd.DataFrame({
        'diff': stat['mean'][a] - stat['mean'][b],
        'diff_interval': interval,
        'relative_diff': (stat['mean'][a] - stat['mean'][b]) / stat['mean'][b],
        'relative_diff_interval': interval / stat['mean'][b],
        a: stat['mean'][a],
        b: stat['mean'][b],
        f'{a}_interval': interval_a,
        f'{b}_interval': interval_b,
        f'{a}_samples': stat['samples'][a],
        f'{b}_samples': stat['samples'][b],
    })

    return result

def generate_query_string(query):
    query_components = list()
    for key,value in query.items():
        query_components.append(f'{key} == {repr(value)}')

    query = ' and '.join(query_components)
    return query

def format_bs(bs):
    (bs,unit) = convert_units(bs)
    return f'{bs:.0f} {unit}'

def plot(axes, result, field, query, index):
    data = result[[field, f"{field}_interval"]]\
        .reset_index()\
        .query(generate_query_string(query))\
        .pivot(index=[index], columns=get_columns(index))

    ax = data[field]\
        .plot.bar(ax=axes, yerr=data[f"{field}_interval"], capsize=1.5, error_kw={'elinewidth':0.5}, edgecolor='black', lw=0.5, color=colors)

    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x) for x in data.index]))
    ax.axhline(0, color='black', lw=0.5, label='_nolegend_')
    #ax.set_title(f"qd {qd}, {workload}")
    #ax.set_xlabel(f"Queue Depth {qd}")
    ax.set_xlabel("")
    #ax.set_ylabel(f"{workload}")
    #ax.legend([workload], loc='best')
    ax.legend().remove()
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='both')
    #ax.xaxis.set_label_position('top')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def plot_throughput(axes, frame, base, new):
    result = calculate_difference(frame, new, base)

    data = result[[base, new]]\
        .reset_index()\
        .pivot(index=['bs'], columns=get_columns('bs'))\
        .sort_index(axis=1, level='qd')

    error = result[[f'{base}_interval', f'{new}_interval']]\
        .reset_index()\
        .pivot(index=['bs'], columns=get_columns('bs'))\
        .sort_index(axis=1, level='qd').to_numpy().transpose()

    ax = data \
        .plot\
        .bar(
            ax=axes,
            #yerr=error,
            logy=True,
            capsize=1.5, error_kw={'elinewidth':0.5}, edgecolor='black', lw=0.5,
            color=colors
        )
    bars = ax.patches
    groups = len(data.index)
    for (bar, hatch) in zip(bars, [None]*groups*4 + ['//']*groups*4):
        if hatch != None:
            bar.set_hatch(hatch)

    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x) for x in data.index]))

    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x) for x in data.index]))
    ax.axhline(0, color='black', lw=0.5, label='_nolegend_')
    ax.set_xlabel("")
    ax.legend().remove()
    ax.set_axisbelow(True)
    #ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='both')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def plot_rnull(frame, field, base = None, new = None, title = 'Comparison'):
    result = calculate_difference(frame, new, base)

    fig, axes = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(13,6))
    fig.suptitle(title)

    plot(axes[0][0], result, field, {'workload':  'randread', 'qd':   1}, 'bs')
    plot(axes[0][1], result, field, {'workload':  'randread', 'qd':   8}, 'bs')
    plot(axes[0][2], result, field, {'workload':  'randread', 'qd':  32}, 'bs')
    plot(axes[0][3], result, field, {'workload':  'randread', 'qd': 128}, 'bs')
    plot(axes[1][0], result, field, {'workload': 'randwrite', 'qd':   1}, 'bs')
    plot(axes[1][1], result, field, {'workload': 'randwrite', 'qd':   8}, 'bs')
    plot(axes[1][2], result, field, {'workload': 'randwrite', 'qd':  32}, 'bs')
    plot(axes[1][3], result, field, {'workload': 'randwrite', 'qd': 128}, 'bs')

    axes[0][0].set_ylabel("randread")
    axes[1][0].set_ylabel("randwrite")
    axes[0][0].set_title("qd 1")
    axes[0][1].set_title("qd 8")
    axes[0][2].set_title("qd 32")
    axes[0][3].set_title("qd 128")
    fig.text(0.01, 0.5, 'IO/s Difference Relative', va='center', rotation='vertical')
    fig.text(0.5, 0.01, 'Block Size (KiB)', ha='center')
    fig.legend(['1', '2', '6'], loc='lower left', ncols=3, title='Threads (cores)', bbox_to_anchor=(0.03,0.85))
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.08, top=0.8)

    print("Mean of difference: {:.3}".format(result[field].mean()))
    print("Samples {}: {:.3}".format(base, result[f'{base}_samples'].mean()))
    print("Samples {}: {:.3}".format(new, result[f'{new}_samples'].mean()))

def violin(ax, frame, base, new, workload, qd):
    blocksizes = [4096, 32768, 262144, 1048576, 16777216]
    jobcounts = [1, 2, 6]
    positions = [1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19]
    query = {'workload':  workload, 'qd': qd, 'config': base}
    frame_c = frame.query(generate_query_string(query))
    query = {'workload':  workload, 'qd': qd, 'config': new}
    frame_r = frame.query(generate_query_string(query))
    data_c = list()
    data_r = list()
    for bs in blocksizes:
        for jc in jobcounts:
            query = {'bs': bs, 'jobcount': jc}
            col_c = frame_c.query(generate_query_string(query))['iops']
            col_r = frame_r.query(generate_query_string(query))['iops']
            mean = col_c.mean()
            col_c /= mean
            col_r /= mean
            data_c.append(col_c)
            data_r.append(col_r)
    side = 'both'
    width = 0.8
    parts_c = ax.violinplot(data_c, side='low', showextrema=True, widths=width, positions=positions)
    parts_r = ax.violinplot(data_r, side='high', showextrema=True, widths=width, positions=positions)

    for i,pc in enumerate(parts_c['bodies']):
        pc.set_facecolor(colors[i%3])
        pc.set_edgecolor(colors[i%3])
        #pc.set_linewidth(2)

    for i,pc in enumerate(parts_r['bodies']):
        pc.set_facecolor(colors[i%3])
        pc.set_edgecolor(colors[i%3])
        #pc.set_linewidth(2)
        pc.set_hatch('X+*')
        pc.set_alpha(0.5)

    parts_c['cbars'].set_linewidth(0.5)
    parts_c['cmaxes'].set_linewidth(0.5)
    parts_c['cmins'].set_linewidth(0.5)
    parts_r['cbars'].set_linewidth(0.5)
    parts_r['cmaxes'].set_linewidth(0.5)
    parts_r['cmins'].set_linewidth(0.5)

    ax.vlines([4,8,12,16], 0, 1,  transform=ax.get_xaxis_transform())

    ax.xaxis.set_major_locator(ticker.FixedLocator([2,6,10,14,18]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x) for x in blocksizes]))
    ax.set_xlabel("")
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #ax.yaxis.grid(True, which='both')
    #ax.xaxis.grid(True, which='both')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def plot_null_violin(frame, base = None, new = None, title = 'Normalized Density'):
    fig, axes = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(13,6))
    fig.suptitle(title)

    violin(axes[0][0], frame, base, new, 'randread', 1)
    violin(axes[0][1], frame, base, new, 'randread', 8)
    violin(axes[0][2], frame, base, new, 'randread', 32)
    violin(axes[0][3], frame, base, new, 'randread', 128)
    violin(axes[1][0], frame, base, new, 'randwrite', 1)
    violin(axes[1][1], frame, base, new, 'randwrite', 8)
    violin(axes[1][2], frame, base, new, 'randwrite', 32)
    violin(axes[1][3], frame, base, new, 'randwrite', 128)

    axes[0][0].set_ylabel("randread")
    axes[1][0].set_ylabel("randwrite")
    axes[0][0].set_title("qd 1")
    axes[0][1].set_title("qd 8")
    axes[0][2].set_title("qd 32")
    axes[0][3].set_title("qd 128")
    fig.text(0.01, 0.5, 'IO/s Relative', va='center', rotation='vertical')
    fig.text(0.5, 0.01, 'Block Size (KiB)', ha='center')
    fig.legend(['1', '2', '6'], loc='lower left', ncols=3, title='Threads (cores)', bbox_to_anchor=(0.03,0.85))
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.08, top=0.8)

def plot_nvme_relative(frame, field, base = None, new = None, title = 'Comparison'):
    result = calculate_difference(frame, new, base)
    fig,axes = plt.subplots(figsize=(6,5))
    fig.suptitle(title)
    plot(axes, result, field, {'workload': 'randread', 'jobcount': 1}, 'bs')

    axes.set_ylabel('Relative difference')
    axes.set_xlabel('Block size')
    fig.legend(['1', '8', '32', '128'], title = 'Queue depth')

    plt.subplots_adjust(bottom=0.2)

    print("Mean of difference: {:.3}".format(result[field].mean()))
    print("Samples {}: {:.3}".format(base, result[f'{base}_samples'].mean()))
    print("Samples {}: {:.3}".format(new, result[f'{new}_samples'].mean()))

def plot_nvme_absolute(frame, base=None, new=None):
    fig,axes = plt.subplots(figsize=(8,5))
    plot_throughput(axes, frame, base, new)
    axes.legend(['C, 1', 'Rust, 1', 'C, 8', 'Rust, 8', 'C, 32', 'Rust, 32', 'C, 128', 'Rust, 128'], title="Configuration (lang, QD)")
    axes.set_title("Random read throughput (Bare Metal, 1 core)")
    axes.set_ylabel("IO/s")
    axes.set_xlabel("Block size")
    plt.subplots_adjust(bottom=0.2)

def nvme_quick(version):
    frame = pd.DataFrame()
    frame = append_single(frame, f'data-rnvme-{version}', 'rust')
    frame = append_single(frame, f'data-rnvme-{version}', 'c')

    plot_nvme_relative(frame, 'relative_diff', 'c', 'rust', r"NVMe randread, $\frac{R-C}{C}$ (Bare Metal, 1 core)")
    #plt.show()
    plt.savefig(f"nvme-{version}-relative.svg")

    plot_nvme_absolute(frame, base='c', new='rust')
    #plt.show()
    plt.savefig(f"nvme-{version}-absolute.svg")

def null_cli(path_a, path_b, name_a, name_b, out_path, out_name):
    frame = pd.DataFrame()
    frame = append_single(frame, path_a, name_a)
    frame = append_single(frame, path_b, name_b)

    plot_rnull(frame, 'relative_diff',  base = name_a, new=name_b, title=r"Null Blk Throughput, $\frac{B-A}{A}$ (Bare Metal)")
    plt.savefig(f'{out_path}/{out_name}.svg')
    plot_null_violin(frame, name_a, name_b)
    plt.savefig(f'{out_path}/{out_name}-density.svg')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-a", required=True)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--path-b", required=True)
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--out-path", default=".")
    parser.add_argument("--out-name", default="plot")
    args = parser.parse_args()

    null_cli(args.path_a, args.path_b, args.name_a, args.name_b, args.out_path, args.out_name)

if __name__ == "__main__":
    main()
