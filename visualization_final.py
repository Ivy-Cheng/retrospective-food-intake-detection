#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def compute_auc(df, trend_col='medtronic_trend_20m_num'):
    df2 = df.copy()
    df2['sign'] = np.sign(df2[trend_col])
    df2['group'] = (df2['sign'] != df2['sign'].shift()).cumsum()
    records = []
    for _, grp in df2.groupby('group'):
        start = grp.index[0]
        end = grp.index[-1]
        length = len(grp)
        sign0 = grp['sign'].iloc[0]
        if pd.isna(sign0):
            auc = np.nan
        elif sign0 == 0:
            auc = 0.0
        else:
            y = grp[trend_col].values
            auc = ((y[:-1] + y[1:]) * 0.5).sum()
        sign_val = int(sign0) if not pd.isna(sign0) else np.nan
        records.append({'start': start, 'end': end, 'length': length, 'auc': auc, 'sign': sign_val})
    return pd.DataFrame.from_records(records)

def create_basic_interactive_trend_figure(df, features, event_features=None):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for feat in features:
        display_name = 'glucose' if feat == 'value' else feat
        fig.add_trace(go.Scattergl(x=df.index, y=df[feat], mode='lines', name=display_name, visible=True))
    if event_features:
        line_feats = [f for f in features if f not in set(event_features)]
        if len(line_feats) > 0:
            ymin = df[line_feats].min().min()
            ymax = df[line_feats].max().max()
        else:
            ymin = df[features].min().min()
            ymax = df[features].max().max()
        event_styles = {'dose': dict(color='green', dash='dot')}
        for ev in event_features:
            series = df[ev].dropna()
            style = event_styles.get(ev, dict(color='gray', dash='dash'))
            xs, ys, texts = [], [], []
            for ts, v in series.items():
                if xs:
                    xs.append(ts); ys.append(None); texts.append(None)
                xs.extend([ts, ts]); ys.extend([ymin, ymax]); texts.extend([v, v])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', line=style,
                name=('bolus' if ev == 'dose' else ev), showlegend=True,
                connectgaps=False, visible=True, text=texts,
                hovertemplate=('time: %{x}<br>' 'bolus: %{text}<extra></extra>')
            ))
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0), margin=dict(t=80))
    # --- Scheme B: make updatemenus follow the selected day (via slider steps) ---
    # Build per-day window buttons and attach them via each slider step's relayout args.
    dates = sorted(set(df.index.date))
    def make_buttons_for_day(day_ts: pd.Timestamp):
        btns = []
        # Whole day
        btns.append(dict(
            label='Whole day',
            method='relayout',
            args=[{'xaxis.range': [day_ts, day_ts + pd.Timedelta(days=1)]}]
        ))
        # 6-hour windows
        for i, (h0, h1) in enumerate([(0,6), (6,12), (12,18), (18,24)], start=1):
            w0 = day_ts + pd.Timedelta(hours=h0)
            w1 = day_ts + pd.Timedelta(hours=h1)
            btns.append(dict(
                label=f'P{i}',
                method='relayout',
                args=[{'xaxis.range': [w0, w1]}]
            ))
        # wrap into one updatemenus config
        menu = dict(
            type='buttons',
            showactive=True,
            direction='right',
            x=0.1, y=1.14,
            xanchor='left', yanchor='top',
            buttons=btns
        )
        return menu

    # Build slider steps: each step sets both xaxis.range (for that day) and the day's updatemenus.
    steps = []
    for d in dates:
        start_day = pd.Timestamp(d)
        end_day = start_day + pd.Timedelta(days=1)
        menu_for_day = make_buttons_for_day(start_day)
        steps.append(dict(
            label=str(d),
            method='relayout',
            args=[{
                'xaxis.range': [start_day, end_day],
                'updatemenus': [menu_for_day],
            }]
        ))
    # Initialize with the first day's buttons so page load状态一致
    initial_menu = make_buttons_for_day(pd.Timestamp(dates[0])) if dates else dict(type='buttons', buttons=[])
    fig.update_layout(
        sliders=[dict(active=0, steps=steps, pad={'t': 50})],
        updatemenus=[initial_menu]
    )
    return fig

def create_wider_figure(df, features, event_features=None):
    """
    Same as create_basic_interactive_trend_figure, but places the day slider
    on the top row above the plot.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for feat in features:
        display_name = 'glucose' if feat == 'value' else feat
        fig.add_trace(go.Scattergl(x=df.index, y=df[feat], mode='lines', name=display_name, visible=True, 
        hovertemplate= ('time: %{x|%H:%M}<br>' 'glucose: %{y:.2f}<extra></extra>')))
    if event_features:
        line_feats = [f for f in features if f not in set(event_features)]
        if len(line_feats) > 0:
            ymin = df[line_feats].min().min()
            ymax = df[line_feats].max().max()
        else:
            ymin = df[features].min().min()
            ymax = df[features].max().max()
        event_styles = {'dose': dict(color='green', dash='dot')}
        for ev in event_features:
            series = df[ev].dropna()
            style = event_styles.get(ev, dict(color='gray', dash='dash'))
            if ev == 'dose':
                # 用绿色的小三角，位于底部上方 2%
                ybase = ymin + (ymax - ymin) * 0.02
                xs = list(series.index)
                ys = [ybase] * len(xs)
                texts = list(series.values)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=10),
                    name='bolus', showlegend=True, visible=True, text=texts,
                    hovertemplate=('time: %{x|%H:%M}<br>' 'bolus: %{text}<extra></extra>')
                ))
            else:
                # 其余事件保持竖线
                xs, ys, texts = [], [], []
                for ts, v in series.items():
                    if xs:
                        xs.append(ts); ys.append(None); texts.append(None)
                    xs.extend([ts, ts]); ys.extend([ymin, ymax]); texts.extend([v, v])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode='lines', line=style,
                    name=ev, showlegend=True,
                    connectgaps=False, visible=True, text=texts,
                    hovertemplate=('time: %{x|%H:%M}<br>' 'bolus: %{text}<extra></extra>')
                ))
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0), margin=dict(t=120))

    # 顶部控件改为外部注入（原生下拉 + Prev/Next），此处不再使用 Plotly 内置下拉
    dates = sorted(set(df.index.date))
    # 顶部留白为外部注入的工具栏预留空间
    fig.update_layout(margin=dict(t=0))
    fig.update_layout(
    paper_bgcolor="white",  # 画布（整个图）背景
    plot_bgcolor="white",   # 绘图区背景
    )
    # 仅底部刻度；默认显示第一天
    if dates:
        first = pd.Timestamp(dates[0])
        fig.update_xaxes(range=[first, first + pd.Timedelta(days=1)])
    fig.update_xaxes(ticks='outside', showline=True, mirror=True, linecolor="black", linewidth=1, showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, mirror=True, linecolor="black", linewidth=1)

    # 生成每个整点的竖线，按“主刻(06/12/18)实线、其余虚线”分两组
    start_day = pd.to_datetime(df.index.min()).floor("D")
    end_day   = (pd.to_datetime(df.index.max()).floor("D") + pd.Timedelta(days=1))
    hours = pd.date_range(start_day, end_day, freq="1H")
    xs_major, ys_major = [], []
    xs_minor, ys_minor = [], []
    for ts in hours:
        # 分割线段（非第一条前加 None 断开）
        if xs_major or xs_minor:
            # 插入分段用的 None：这两组都各自断开
            xs_major.append(None); ys_major.append(None)
            xs_minor.append(None); ys_minor.append(None)
        if ts.hour in (6, 12, 18):
            xs_major.extend([ts, ts]); ys_major.extend([ymin, ymax])
        else:
            xs_minor.extend([ts, ts]); ys_minor.extend([ymin, ymax])
    # 两条虚线
    fig.add_trace(go.Scatter(
        x=xs_minor, y=ys_minor, mode="lines",
        line=dict(color="rgba(0,0,0,0.18)", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False, name="_hour_minor"
    ))
    fig.add_trace(go.Scatter(
        x=xs_major, y=ys_major, mode="lines",
        line=dict(color="rgba(0,0,0,0.28)", width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False, name="_hour_major"
    ))
  
    return fig


def write_html_with_prev_next(fig, dates, out_path):
    """
    将 fig 写为单一 HTML，并注入 Prev/Next 控件（纯静态可用）。
    dates: 可迭代的日期字符串或 date 对象（会被 str()）
    """
    if dates:
        first = pd.Timestamp(str(dates[0]))
        fig.update_xaxes(range=[first, first + pd.Timedelta(days=1)])
    js_dates = [str(d) for d in dates]
    # 使用 post_script 注入，定位第一个 plotly graph div
    post_script = (
        "(function(){\n"
        "  var gd = document.querySelectorAll('div.plotly-graph-div')[0];\n"
        f"  var dts = {js_dates!r};\n"
        "  var cur = 0;\n"
        "  function applyRange(){\n"
        "    var start = new Date(dts[cur]);\n"
        "    var end = new Date(+start + 24*60*60*1000);\n"
        "    Plotly.relayout(gd, {'xaxis.range':[start, end]});\n"
        "  }\n"
        "  function setDay(i, fromSelect){\n"
        "    if (i < 0){ alert('No previous day'); return; }\n"
        "    if (i >= dts.length){ alert('No next day'); return; }\n"
        "    cur = i;\n"
        "    applyRange();\n"
        "    if (!fromSelect) sel.value = String(cur);\n"
        "  }\n"
        "  // 构建顶栏：Day: [select] [Prev] [Next] [Help?]\n"
        "  var bar = document.createElement('div');\n"
        "  bar.style.cssText = 'position:relative; margin:4px 0 8px; display:flex; align-items:center; gap:8px;';\n"
        "  var label = document.createElement('span'); label.textContent = 'Day:';\n"
        "  var sel = document.createElement('select');\n"
        "  for (var i=0;i<dts.length;i++){\n"
        "    var opt = document.createElement('option');\n"
        "    opt.value = String(i);\n"
        "    opt.textContent = dts[i];\n"
        "    sel.appendChild(opt);\n"
        "  }\n"
        "  var prev = document.createElement('button'); prev.textContent = 'Prev';\n"
        "  var next = document.createElement('button'); next.textContent = 'Next';\n"
        "  prev.onclick = function(){ setDay(cur-1, false); };\n"
        "  next.onclick = function(){ setDay(cur+1, false); };\n"
        "  sel.onchange = function(){ var idx = parseInt(sel.value||'0',10); setDay(idx, true); };\n"
        "  var helpBtn = document.createElement('button'); helpBtn.type = 'button'; helpBtn.textContent = 'Help?';\n"
        "  helpBtn.onclick = function(){\n"
        "    if (typeof window.__faOpenHelp === 'function') { window.__faOpenHelp(); return; }\n"
        "    var mask = document.getElementById('fa-help-mask');\n"
        "    var dlg = document.getElementById('fa-help-dialog');\n"
        "    if (!mask || !dlg) return;\n"
        "    mask.style.display = 'block'; dlg.style.display = 'block';\n"
        "  };\n"
        "  bar.appendChild(label); bar.appendChild(sel); bar.appendChild(prev); bar.appendChild(next); bar.appendChild(helpBtn);\n"
        "  var host = gd.parentNode; host.insertBefore(bar, host.firstChild);\n"
        "  // 默认第一天\n"
        "  setDay(0, true);\n"
        "})();"
    )
    html = fig.to_html(full_html=True, include_plotlyjs='cdn', post_script=post_script)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding='utf-8')

def create_interactive_trend_figure(df, features, event_features=None, result_timestamps=None, ra_series=None, ra_name='ra_estimate'):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    result_styles = {
        'simple_heuristic': {'y': 0.9, 'color': 'purple', 'symbol': 'circle', 'size': 10},
        'trend': {'y': 0.7, 'color': 'blue', 'symbol': 'square', 'size': 10},
        'LSTM_cAE': {'y': 0.5, 'color': 'green', 'symbol': 'triangle-up', 'size': 10},
        'fusion_based': {'y': 0.3, 'color': 'red', 'symbol': 'triangle-down', 'size': 10},
        'physio_kf': {'y': 0.1, 'color': 'yellow', 'symbol': 'diamond', 'size': 10}
    }
    line_feats = [f for f in features if f not in event_features]
    trend_feats = [f for f in line_feats if '_trend_' in f]
    non_trend_feats = [f for f in line_feats if '_trend_' not in f]
    ymin = df[line_feats].min().min(); ymax = df[line_feats].max().max()
    trace_indices = {}; current_index = 0
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{}]], shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.05)
    for feat in non_trend_feats:
        fig.add_trace(go.Scattergl(x=df.index, y=df[feat], mode='lines', name=('glucose' if feat == 'value' else feat), visible=True), row=1, col=1)
        trace_indices[feat] = current_index; current_index += 1
    for feat in trend_feats:
        start_index = current_index
        fig.add_trace(go.Scattergl(x=df.index, y=df[feat], mode='lines', name=('glucose' if feat == 'value' else feat), visible=True), row=1, col=1)
        current_index += 1
        auc_df = compute_auc(df, feat)
        for _, row in auc_df.iterrows():
            if pd.isna(row['sign']) or row['sign'] == 0: continue
            mask = (df.index >= row['start']) & (df.index <= row['end'])
            segment = df[mask]
            fillcolor = 'rgba(255,0,0,0.1)' if row['sign'] > 0 else 'rgba(0,0,255,0.1)'
            fig.add_trace(go.Scatter(x=segment.index, y=segment[feat], fill='tozeroy', fillcolor=fillcolor, line=dict(width=0),
                                     showlegend=False, name=f'AUC: {row["auc"]:.2f}', hoverinfo='name', visible=True), row=1, col=1)
            current_index += 1
        trace_indices[feat] = list(range(start_index, current_index))
    event_styles = {'carbs': dict(color='orange', dash='dash'), 'dose': dict(color='green', dash='dot')}
    for ev in event_features:
        series = df[ev].dropna(); style = event_styles.get(ev, dict(color='gray', dash='dash'))
        xs = []; ys = []
        for ts, v in series.items():
            if xs: xs.append(ts); ys.append(None)
            xs.extend([ts, ts]); ys.extend([ymin, ymax])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=style, name=('bolus' if ev == 'dose' else ev),
                                 showlegend=True, hoverinfo='skip', connectgaps=False, visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=style, name=ev + '_lower',
                                 showlegend=False, hoverinfo='skip', connectgaps=False, visible=True), row=2, col=1)
        trace_indices[ev] = [current_index, current_index + 1, current_index + 2]; current_index += 3
    if ra_series is not None:
        rs = ra_series.copy()
        if not isinstance(rs.index, pd.DatetimeIndex): rs.index = pd.to_datetime(rs.index)
        rs = rs.reindex(df.index)
        fig.add_trace(go.Scattergl(x=df.index, y=rs, mode='lines', name=ra_name, line=dict(color='crimson'), visible=True), row=1, col=1, secondary_y=True)
        trace_indices[ra_name] = current_index; current_index += 1
        fig.update_yaxes(title_text="Ra (mmol/L/min)", row=1, col=1, secondary_y=True)
    if result_timestamps:
        for result_type, timestamps in result_timestamps.items():
            if result_type in result_styles and timestamps:
                style = result_styles[result_type]; timestamps = pd.to_datetime(timestamps)
                fig.add_trace(go.Scatter(x=timestamps, y=[style['y']] * len(timestamps), mode='markers',
                                         marker=dict(symbol=style['symbol'], size=style['size'], color=style['color']),
                                         name=f'{result_type.replace("_", " ").title()} Result', showlegend=True, hoverinfo='x', visible=True), row=2, col=1)
                trace_indices[f'{result_type}_result'] = current_index; current_index += 1
    # Feature buttons (original)
    all_buttons = []
    for feat in non_trend_feats:
        all_buttons.append(dict(label=feat, method='restyle', args=[{'visible': [False]}, [trace_indices[feat]]], args2=[{'visible': [True]}, [trace_indices[feat]]]))
    for feat in trend_feats:
        all_buttons.append(dict(label=feat, method='restyle', args=[{'visible': [False]}, trace_indices[feat]], args2=[{'visible': [True]}, trace_indices[feat]]))
    for ev in event_features:
        all_buttons.append(dict(label=ev, method='restyle', args=[{'visible': [False]}, trace_indices[ev]], args2=[{'visible': [True]}, trace_indices[ev]]))
    if ra_series is not None:
        all_buttons.append(dict(label=ra_name, method='restyle', args=[{'visible': [False]}, [trace_indices[ra_name]]], args2=[{'visible': [True]}, [trace_indices[ra_name]]]))
    if result_timestamps:
        for result_type in result_timestamps:
            if result_type in result_styles:
                all_buttons.append(dict(label=f'{result_type.replace("_", " ").title()} Result', method='restyle',
                                        args=[{'visible': [False]}, [trace_indices[f'{result_type}_result']]], args2=[{'visible': [True]}, [trace_indices[f'{result_type}_result']]]))
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=True, direction='right', x=0.1, y=1.1, xanchor='left', yanchor='top', buttons=all_buttons)])
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    # Day slider unchanged for interactive
    dates = sorted(set(df.index.date))
    steps = []
    for d in dates:
        start = pd.Timestamp(d); end = start + pd.Timedelta(days=1)
        steps.append(dict(label=str(d), method='relayout', args=[{'xaxis.range': [start, end]}]))
    fig.update_layout(sliders=[dict(active=0, steps=steps, pad={'t':50})])
    return fig
