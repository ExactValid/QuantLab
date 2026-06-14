# scripts/strategy/html_renderer.py
import json
from pathlib import Path

def generate_human_dashboard(dataset_name):
    target_folder = Path("outputs/records") / dataset_name
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if not target_folder.exists(): return
        
    json_files = sorted(list(target_folder.glob("*.json")))
    merged_ledger = {}
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 🌟 用模型唯一ID作为第一层主键，阻断覆盖
                merged_ledger[file_path.stem] = data
        except Exception: continue
            
    if not merged_ledger: return
    
    html_top = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>量化投研专家级全景看板 - __DATASET_NAME__</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f4f8; color: #1e293b; margin: 0; padding: 30px; }
        h1 { color: #0f172a; border-bottom: 3px solid #cbd5e1; padding-bottom: 12px; font-size: 26px; font-weight: 700; }
        .matrix-card { background: #fff; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); border-radius: 16px; padding: 28px; margin-bottom: 40px; border: 1px solid #e2e8f0; }
        .model-header { border-left: 6px solid #e11d48; padding-left: 18px; margin-bottom: 25px; }
        .model-title { font-size: 20px; font-weight: 700; color: #0f172a; font-family: monospace; }
        .model-meta { font-size: 13px; color: #64748b; margin-top: 6px; }
        .grid-layout { display: flex; flex-direction: column; gap: 25px; }
        @media(min-width: 1200px) { .grid-layout { flex-direction: row; } }
        .table-area { flex: 4; }
        .charts-container { flex: 5; display: flex; flex-direction: column; gap: 15px; }
        .chart-box { background: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0; padding: 15px; min-height: 200px; }
        table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
        th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 13px; }
        th { background-color: #1e293b; color: #f8fafc; font-weight: 600; font-size: 11px; text-transform: uppercase; }
        .opt-name { font-weight: 700; color: #2563eb; font-size: 14px; }
        .view-btn { background: #3b82f6; color: white; border: none; padding: 4px 8px; font-size: 11px; border-radius: 4px; cursor: pointer; margin-left: 8px; font-weight: bold; }
        .view-btn:hover { background: #1d4ed8; }
        .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .metric-label { color: #64748b; font-size: 12px; }
        .metric-val { font-weight: 600; color: #0f172a; }
        .negative { color: #dc2626 !important; }
        .positive { color: #16a34a !important; }
        .audit-section { margin-top: 30px; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; }
        .audit-title { font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }
        .audit-wrapper { max-height: 350px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; }
    </style>
</head>
<body>
    <h1>量化投研全景看板</h1>
"""
    html_content = html_top.replace("__DATASET_NAME__", dataset_name)
    
    js_data_registry = {}
    chart_init_scripts = ""
    
    for m_idx, (model_id, model_data) in enumerate(merged_ledger.items()):
        meta = model_data.get("metadata", {})
        val_perf = model_data.get("val_performance", {})
        backtest = model_data.get("backtest_records", {})
        
        js_data_registry[m_idx] = {}
        default_opt = None
        
        equity_chart_id = f"equity_chart_{m_idx}"
        pos_chart_id = f"pos_chart_{m_idx}"  
        turn_chart_id = f"turn_chart_{m_idx}" 
        dd_chart_id = f"dd_chart_{m_idx}"    
        tbody_id = f"audit_tbody_{m_idx}"
        title_id = f"active_strategy_title_{m_idx}"
        
        # 不同的算法模型给予不同的主色调边框
        border_color = "#3b82f6" if "GradientBoosting" in model_id else "#10b981"
        
        html_content += f"""
        <div class="matrix-card" style="border-top: 6px solid {border_color};">
            <div class="model-header" style="border-left: 6px solid {border_color};">
                <div class="model-title">{model_id}</div>
                <div class="model-meta">
                    验证集准确率: <span class="metric-val" style="color:{border_color}; font-size:15px;">{val_perf.get('accuracy', 'N/A')}</span> | 
                    数据源: {meta.get('source_dataset', 'N/A')}
                </div>
            </div>
            
            <div class="grid-layout">
                <div class="table-area">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 35%;">配置优化策略</th>
                                <th style="width: 65%;">归因指标矩阵 (Risk Attribution Matrix)</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for opt in sorted(list(backtest.keys())):
            rec = backtest[opt]
            if default_opt is None: default_opt = opt
            
            ann_ret = rec.get("annualized_return", "0.0%")
            max_dd = rec.get("max_drawdown", "0.0%")
            sharpe = float(rec.get("sharpe_ratio", 0.0))
            win_rate = rec.get("win_rate", "N/A")
            profit_loss = rec.get("profit_loss_ratio", "N/A")
            ret_style = "negative" if "-" in ann_ret else "positive"
            
            mock_dates = rec["date_series"]
            js_data_registry[m_idx][opt] = {
                "dates": mock_dates,
                "equity": rec["equity_series"],
                "drawdown": rec["drawdown_series"],
                "probs": rec.get("prob_series", [0.5] * len(mock_dates)),
                "positions": rec.get("position_series", [0.0] * len(mock_dates)),
                "turnovers": rec.get("turnover_series", [0.0] * len(mock_dates))
            }
            
            html_content += f"""
            <tr>
                <td>
                    <span class="opt-name">{opt}</span>
                    <button class="view-btn" onclick="switchPerspective({m_idx}, '{opt}')">🔎 穿透透视</button>
                </td>
                <td>
                    <div class="metrics-grid">
                        <div class="metric-item"><span class="metric-label">年化收益:</span> <span class="metric-val {ret_style}">{ann_ret}</span></div>
                        <div class="metric-item"><span class="metric-label">最大回撤:</span> <span class="metric-val negative">{max_dd}</span></div>
                        <div class="metric-item"><span class="metric-label">夏普比率(Sharpe):</span> <span class="metric-val">{sharpe}</span></div>
                        <div class="metric-item"><span class="metric-label">卡玛比率(Calmar):</span> <span class="metric-val positive">{round(float(ann_ret.replace("%",""))/abs(float(max_dd.replace("%",""))),2) if float(max_dd.replace("%",""))!=0 else "N/A"}</span></div>
                        <div class="metric-item"><span class="metric-label">真实交易胜率:</span> <span class="metric-val" style="color:#2563eb;">{win_rate}</span></div>
                        <div class="metric-item"><span class="metric-label">微观盈亏比:</span> <span class="metric-val" style="color:#d97706;">{profit_loss}</span></div>
                    </div>
                </td>
            </tr>
            """
        
        html_content += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="charts-container">
                    <div class="chart-box" id="{equity_chart_id}"></div>
                    <div class="chart-box" id="{pos_chart_id}"></div>
                    <div class="chart-box" id="{turn_chart_id}"></div>
                    <div class="chart-box" id="{dd_chart_id}"></div>
                </div>
            </div>
            
            <div class="audit-section">
                <div class="audit-title">
                    <span> 策略微观日度动态仓位审计账本 (<span id="{title_id}">{default_opt}</span>)</span>
                </div>
                <div class="audit-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 20%;">交易结算日期</th>
                                <th style="width: 25%;">真实策略累计净值</th>
                                <th style="width: 20%;">模型预测多头概率</th>
                                <th style="width: 20%;">真实控仓权重 (Weights)</th>
                                <th style="width: 15%;">单日换手率 (Turnover)</th>
                            </tr>
                        </thead>
                        <tbody id="{tbody_id}"></tbody>
                    </table>
                </div>
            </div>
        </div>
        """
        
        if default_opt is not None:
            def_data = js_data_registry[m_idx][default_opt]
            js_block = """
            window.charts_eq = window.charts_eq || {};
            window.charts_pos = window.charts_pos || {};
            window.charts_turn = window.charts_turn || {};
            window.charts_dd = window.charts_dd || {};
            
            var eqChart = echarts.init(document.getElementById('__EQ_ID__'));
            var posChart = echarts.init(document.getElementById('__POS_ID__'));
            var turnChart = echarts.init(document.getElementById('__TURN_ID__'));
            var ddChart = echarts.init(document.getElementById('__DD_ID__'));
            
            window.charts_eq[__M_IDX__] = eqChart;
            window.charts_pos[__M_IDX__] = posChart;
            window.charts_turn[__M_IDX__] = turnChart;
            window.charts_dd[__M_IDX__] = ddChart;
            
            var commonGrid = { top: '15%', bottom: '15%', left: '10%', right: '5%' };
            
            eqChart.setOption({
                title: { text: '真实回测净值走势曲线 (__OPT__)', textStyle: { fontSize: 13, color: '#334155' }, left: 'center' },
                tooltip: { trigger: 'axis' }, grid: commonGrid,
                xAxis: { type: 'category', data: __DATES__ }, yAxis: { type: 'value', min: 'dataMin' },
                dataZoom: [{ type: 'inside', xAxisIndex: [0] }],
                series: [{ type: 'line', data: __EQUITY__, smooth: true, lineStyle: { width: 2, color: '#2563eb' }, showSymbol: false }]
            });
            
            posChart.setOption({
                title: { text: '历史真实持仓仓位动态波动', textStyle: { fontSize: 13, color: '#334155' }, left: 'center' },
                tooltip: { trigger: 'axis', formatter: function(p){ return p[0].name + '<br/>真实持仓: ' + (p[0].value * 100).toFixed(1) + '%'; } }, grid: commonGrid,
                xAxis: { type: 'category', data: __DATES__ }, yAxis: { type: 'value', min: 0, max: 1 },
                dataZoom: [{ type: 'inside', xAxisIndex: [0] }],
                series: [{ type: 'line', data: __POSITIONS__, smooth: true, lineStyle: { width: 1.5, color: '#0ea5e9' }, areaStyle: { color: 'rgba(14, 165, 233, 0.2)' }, showSymbol: false }]
            });
            
            turnChart.setOption({
                title: { text: '日度交易换手率 (Turnover Rate)', textStyle: { fontSize: 13, color: '#334155' }, left: 'center' },
                tooltip: { trigger: 'axis', formatter: function(p){ return p[0].name + '<br/>单日换手: ' + (p[0].value * 100).toFixed(1) + '%'; } }, grid: commonGrid,
                xAxis: { type: 'category', data: __DATES__ }, yAxis: { type: 'value' },
                dataZoom: [{ type: 'inside', xAxisIndex: [0] }],
                series: [{ type: 'bar', data: __TURNOVERS__, itemStyle: { color: '#d97706' } }]
            });
            
            ddChart.setOption({
                title: { text: '水下动态回撤深度分布', textStyle: { fontSize: 13, color: '#334155' }, left: 'center' },
                tooltip: { trigger: 'axis', formatter: function(p){ return p[0].name + '<br/>回撤: ' + (p[0].value * 100).toFixed(2) + '%'; } },
                grid: { top: '15%', bottom: '25%', left: '10%', right: '5%' },
                xAxis: { type: 'category', data: __DATES__ }, yAxis: { type: 'value', max: 0 },
                dataZoom: [{ type: 'slider', xAxisIndex: [0], bottom: '5%', height: 20 }],
                series: [{ type: 'line', data: __DRAWDOWN__, smooth: true, lineStyle: { width: 1.5, color: '#e11d48' }, showSymbol: false }]
            });
            
            echarts.connect([eqChart, posChart, turnChart, ddChart]);
            
            function makeListener(m, o) {
                return function() {
                    var axis = ddChart.getOption().xAxis[0];
                    if(axis.rangeStart !== undefined) sliceAuditTable(m, o, axis.rangeStart, axis.rangeEnd);
                }
            }
            ddChart.on('dataZoom', makeListener(__M_IDX__, '__OPT__'));
            eqChart.on('dataZoom', makeListener(__M_IDX__, '__OPT__'));
            
            sliceAuditTable(__M_IDX__, '__OPT__', 0, __DATES__.length - 1);
            """
            js_block = js_block.replace("__EQ_ID__", equity_chart_id)\
                               .replace("__POS_ID__", pos_chart_id)\
                               .replace("__TURN_ID__", turn_chart_id)\
                               .replace("__DD_ID__", dd_chart_id)\
                               .replace("__M_IDX__", str(m_idx))\
                               .replace("__OPT__", default_opt)\
                               .replace("__DATES__", json.dumps(def_data["dates"]))\
                               .replace("__EQUITY__", json.dumps(def_data["equity"]))\
                               .replace("__POSITIONS__", json.dumps(def_data["positions"]))\
                               .replace("__TURNOVERS__", json.dumps(def_data["turnovers"]))\
                               .replace("__DRAWDOWN__", json.dumps(def_data["drawdown"]))
            chart_init_scripts += js_block

    html_content += """
    <script>
        var masterRegistry = """ + json.dumps(js_data_registry) + """;
        window.onload = function() { """ + chart_init_scripts + """ }
        
        function sliceAuditTable(mIdx, optName, startIdx, endIdx) {
            var data = masterRegistry[mIdx][optName];
            if(!data) return;
            var tbody = document.getElementById('audit_tbody_' + mIdx);
            if(!tbody) return;
            var html = '';
            if(startIdx < 0) startIdx = 0;
            if(endIdx >= data.dates.length) endIdx = data.dates.length - 1;
            
            for(var i = startIdx; i <= endIdx; i++) {
                var tColor = data.turnovers[i] > 0.05 ? 'color:#d97706;font-weight:bold;' : 'color:#64748b;';
                html += '<tr>';
                html += '<td><b>' + data.dates[i] + '</b></td>';
                html += '<td>' + data.equity[i].toFixed(4) + '</td>';
                html += '<td>' + (data.probs[i] * 100).toFixed(1) + '%</td>';
                html += '<td><b>' + (data.positions[i] * 100).toFixed(1) + '%</b></td>';
                html += '<td style="' + tColor + '">' + (data.turnovers[i] * 100).toFixed(1) + '%</td>';
                html += '</tr>';
            }
            tbody.innerHTML = html;
        }
        
        function switchPerspective(mIdx, optName) {
            var data = masterRegistry[mIdx][optName];
            if(!data) return;
            document.getElementById('active_strategy_title_' + mIdx).innerText = optName;
            window.charts_eq[mIdx].setOption({ title: { text: '真实回测净值走势曲线 (' + optName + ')' }, xAxis: { data: data.dates }, series: [{ data: data.equity }] });
            window.charts_pos[mIdx].setOption({ xAxis: { data: data.dates }, series: [{ data: data.positions }] });
            window.charts_turn[mIdx].setOption({ xAxis: { data: data.dates }, series: [{ data: data.turnovers }] });
            window.charts_dd[mIdx].setOption({ xAxis: { data: data.dates }, series: [{ data: data.drawdown }] });
            sliceAuditTable(mIdx, optName, 0, data.dates.length - 1);
        }
    </script>
</body>
</html>
    """
    with open(reports_dir / f"Dashboard_{dataset_name}.html", 'w', encoding='utf-8') as f:
        f.write(html_content)