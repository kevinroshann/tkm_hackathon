from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime

LOGS_FILE = "emergency_logs.json"

def create_department_app(department_name, port):
    app = Flask(__name__)
    CORS(app)

    TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ department }} Department - Emergency Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: #2c3e50;
            color: white;
            padding: 30px 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 8px;
        }

        header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }

        header p {
            opacity: 0.9;
            font-size: 1.1em;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }

        .tab {
            padding: 15px 30px;
            background: white;
            border: none;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            transition: all 0.3s;
        }

        .tab.active {
            background: #3498db;
            color: white;
        }

        .tab:hover {
            background: #e8e8e8;
        }

        .tab.active:hover {
            background: #2980b9;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-box {
            background: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }

        .stat-label {
            color: #666;
            text-transform: uppercase;
            font-size: 0.9em;
        }

        .stat-box.critical .stat-number {
            color: #dc3545;
        }

        .stat-box.urgent .stat-number {
            color: #ffc107;
        }

        .stat-box.total .stat-number {
            color: #3498db;
        }

        .stat-box.resolved .stat-number {
            color: #28a745;
        }

        .controls {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            flex-wrap: wrap;
            gap: 15px;
        }

        .filter-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        select {
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            background: white;
        }

        button {
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }

        button:hover {
            background: #2980b9;
        }

        .resolve-btn {
            background: #28a745;
            padding: 8px 16px;
            font-size: 0.9em;
        }

        .resolve-btn:hover {
            background: #218838;
        }

        .resolve-btn.resolved {
            background: #6c757d;
            cursor: not-allowed;
        }

        .auto-refresh-active {
            animation: pulse-button 2s infinite;
        }

        @keyframes pulse-button {
            0%, 100% { 
                box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7);
            }
            50% { 
                box-shadow: 0 0 0 10px rgba(52, 152, 219, 0);
            }
        }

        .logs-grid {
            display: grid;
            gap: 20px;
        }

        .log-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 5px solid #3498db;
            transition: all 0.3s;
        }

        .log-card.resolved {
            background: #d4edda;
            border-left-color: #28a745;
        }

        .log-card.critical {
            border-left-color: #dc3545;
        }

        .log-card.urgent {
            border-left-color: #ffc107;
        }

        .log-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }

        .log-id {
            font-weight: 600;
            color: #2c3e50;
        }

        .timestamp {
            color: #666;
            font-size: 0.9em;
        }

        .priority-badge {
            padding: 6px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
        }

        .priority-badge.CRITICAL {
            background: #dc3545;
            color: white;
        }

        .priority-badge.URGENT {
            background: #ffc107;
            color: #333;
        }

        .priority-badge.STANDARD {
            background: #28a745;
            color: white;
        }

        .sos-badge {
            background: #dc3545;
            color: white;
            padding: 6px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
        }

        .resolved-badge {
            background: #28a745;
            color: white;
            padding: 6px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
        }

        .user-info {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 12px;
        }

        .complaint-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 12px 0;
            border-left: 3px solid #3498db;
        }

        .severity-row {
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 12px 0;
        }

        .severity-score {
            min-width: 80px;
            padding: 8px 15px;
            border-radius: 6px;
            font-weight: 600;
            text-align: center;
        }

        .severity-score.high {
            background: #dc3545;
            color: white;
        }

        .severity-score.medium {
            background: #ffc107;
            color: #333;
        }

        .severity-score.low {
            background: #28a745;
            color: white;
        }

        .action-box {
            background: #e3f2fd;
            padding: 12px;
            border-radius: 6px;
            margin-top: 12px;
            border-left: 3px solid #2196f3;
            font-weight: 500;
        }

        .chart-container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .chart-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2c3e50;
        }

        canvas {
            max-height: 400px;
        }

        .visualization-filters {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .no-logs {
            text-align: center;
            padding: 60px 20px;
            color: #999;
            font-size: 1.2em;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 40px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .stats-row {
                grid-template-columns: repeat(2, 1fr);
            }

            .controls {
                flex-direction: column;
                align-items: stretch;
            }

            .tabs {
                overflow-x: auto;
            }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 {{ department }} Department</h1>
            <p>Emergency Response Dashboard</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('cases')">Cases</button>
            <button class="tab" onclick="switchTab('visualizations')">Visualizations</button>
        </div>

        <!-- CASES TAB -->
        <div id="casesTab" class="tab-content active">
            <div class="stats-row">
                <div class="stat-box total">
                    <div class="stat-label">Total Assigned</div>
                    <div class="stat-number" id="totalCount">0</div>
                </div>
                <div class="stat-box critical">
                    <div class="stat-label">Critical</div>
                    <div class="stat-number" id="criticalCount">0</div>
                </div>
                <div class="stat-box urgent">
                    <div class="stat-label">Urgent</div>
                    <div class="stat-number" id="urgentCount">0</div>
                </div>
                <div class="stat-box resolved">
                    <div class="stat-label">Resolved</div>
                    <div class="stat-number" id="resolvedCount">0</div>
                </div>
            </div>

            <div class="controls">
                <div class="filter-group">
                    <h3 style="margin: 0;">Assigned Complaints</h3>
                    <select id="statusFilter" onchange="displayLogs()">
                        <option value="all">All Cases</option>
                        <option value="active">Active Only</option>
                        <option value="resolved">Resolved Only</option>
                    </select>
                </div>
                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 5px;">
                    <div style="font-size: 0.85em; color: #666;" id="lastUpdated">Last updated: Never</div>
                    <div style="display: flex; gap: 10px;">
                        <button onclick="refreshLogs()">🔄 Refresh</button>
                        <button onclick="toggleAutoRefresh()">⏱️ Auto: <span id="autoStatus">ON</span></button>
                    </div>
                </div>
            </div>

            <div class="logs-grid" id="logsContainer">
                <div class="spinner"></div>
            </div>
        </div>

        <!-- VISUALIZATIONS TAB -->
        <div id="visualizationsTab" class="tab-content">
            <div class="visualization-filters">
                <h3>Filters:</h3>
                <select id="vizTimeRange" onchange="updateCharts()">
                    <option value="all">All Time</option>
                    <option value="today">Today</option>
                    <option value="week">Last 7 Days</option>
                    <option value="month">Last 30 Days</option>
                </select>
                <select id="vizSOSFilter" onchange="updateCharts()">
                    <option value="all">All Cases</option>
                    <option value="sos">SOS Only</option>
                    <option value="regular">Regular Only</option>
                </select>
            </div>

            <div class="chart-container">
                <div class="chart-title">Cases Over Time</div>
                <canvas id="timeChart"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-title">Case Distribution</div>
                <canvas id="distributionChart"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-title">Resolution Status</div>
                <canvas id="resolutionChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const DEPARTMENT = '{{ department }}';
        const API_URL = 'http://localhost:5000';
        let autoRefreshInterval = null;
        let allLogs = [];
        let timeChart, distributionChart, resolutionChart;

        window.addEventListener('load', () => {
            refreshLogs();
            const autoButton = document.querySelector('button[onclick="toggleAutoRefresh()"]');
            autoRefreshInterval = setInterval(refreshLogs, 2000);
            autoButton.classList.add('auto-refresh-active');
            
            initCharts();
        });

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            if (tabName === 'cases') {
                document.getElementById('casesTab').classList.add('active');
                document.querySelectorAll('.tab')[0].classList.add('active');
            } else {
                document.getElementById('visualizationsTab').classList.add('active');
                document.querySelectorAll('.tab')[1].classList.add('active');
                updateCharts();
            }
        }

        async function refreshLogs() {
            try {
                const response = await fetch(`${API_URL}/logs/${DEPARTMENT}`);
                allLogs = await response.json();
                updateStats(allLogs);
                displayLogs();
                
                const now = new Date();
                document.getElementById('lastUpdated').textContent = 
                    `Last updated: ${now.toLocaleTimeString()}`;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('logsContainer').innerHTML = 
                    '<div class="no-logs">Failed to connect to server.</div>';
            }
        }

        function updateStats(logs) {
            const total = logs.length;
            const critical = logs.filter(log => log.priority === 'CRITICAL' && !log.resolved).length;
            const urgent = logs.filter(log => log.priority === 'URGENT' && !log.resolved).length;
            const resolved = logs.filter(log => log.resolved).length;

            document.getElementById('totalCount').textContent = total;
            document.getElementById('criticalCount').textContent = critical;
            document.getElementById('urgentCount').textContent = urgent;
            document.getElementById('resolvedCount').textContent = resolved;
        }

        function displayLogs() {
            const statusFilter = document.getElementById('statusFilter').value;
            let filtered = allLogs;

            if (statusFilter === 'active') {
                filtered = filtered.filter(log => !log.resolved);
            } else if (statusFilter === 'resolved') {
                filtered = filtered.filter(log => log.resolved);
            }

            const container = document.getElementById('logsContainer');

            if (filtered.length === 0) {
                container.innerHTML = '<div class="no-logs">No complaints found matching the filters.</div>';
                return;
            }

            container.innerHTML = filtered.map(log => {
                const severityClass = log.severity > 85 ? 'high' : log.severity > 60 ? 'medium' : 'low';
                const resolvedClass = log.resolved ? 'resolved' : '';
                
                return `
                <div class="log-card ${log.priority.toLowerCase()} ${resolvedClass}">
                    <div class="log-header">
                        <div class="log-id">Complaint #${log.id}</div>
                        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                            ${log.sos ? '<div class="sos-badge">🚨 SOS</div>' : ''}
                            ${log.resolved ? '<div class="resolved-badge">✅ RESOLVED</div>' : ''}
                            <div class="priority-badge ${log.priority}">${log.priority}</div>
                            <div class="timestamp">${formatTimestamp(log.timestamp)}</div>
                        </div>
                    </div>

                    <div class="user-info">
                        <strong>Reported by:</strong> ${log.user_name}<br>
                        <strong>Contact:</strong> ${log.user_contact}
                    </div>

                    <div class="complaint-box">
                        <strong>Description:</strong><br>
                        ${log.complaint}
                    </div>

                    <div class="severity-row">
                        <strong>Severity:</strong>
                        <div class="severity-score ${severityClass}">
                            ${log.severity}/100
                        </div>
                    </div>

                    <div class="action-box">
                        📋 ${log.action}
                    </div>

                    <div style="margin-top: 15px; text-align: right;">
                        ${log.resolved 
                            ? '<button class="resolve-btn resolved" disabled>✅ Resolved</button>'
                            : `<button class="resolve-btn" onclick="resolveCase('${log.id}')">✓ Mark as Resolved</button>`
                        }
                    </div>
                </div>
            `;
            }).join('');
        }

        async function resolveCase(caseId) {
            try {
                const response = await fetch(`${API_URL}/resolve/${caseId}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    const log = allLogs.find(l => l.id === caseId);
                    if (log) {
                        log.resolved = true;
                        log.resolved_at = new Date().toISOString();
                    }
                    updateStats(allLogs);
                    displayLogs();
                    updateCharts();
                }
            } catch (error) {
                console.error('Error resolving case:', error);
                alert('Failed to resolve case. Please try again.');
            }
        }

        function formatTimestamp(timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleString();
        }

        function toggleAutoRefresh() {
            const button = event.target.closest('button');
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
                document.getElementById('autoStatus').textContent = 'OFF';
                button.classList.remove('auto-refresh-active');
            } else {
                autoRefreshInterval = setInterval(refreshLogs, 2000);
                document.getElementById('autoStatus').textContent = 'ON';
                button.classList.add('auto-refresh-active');
                refreshLogs();
            }
        }

        function initCharts() {
            const timeCtx = document.getElementById('timeChart').getContext('2d');
            timeChart = new Chart(timeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Cases Registered',
                        data: [],
                        backgroundColor: '#3498db',
                        borderColor: '#2980b9',
                        borderWidth: 2,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: { legend: { display: true } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { stepSize: 1 }
                        }
                    }
                }
            });

            const distCtx = document.getElementById('distributionChart').getContext('2d');
            distributionChart = new Chart(distCtx, {
                type: 'bar',
                data: {
                    labels: ['Critical', 'Urgent', 'Standard'],
                    datasets: [{
                        label: 'Number of Cases',
                        data: [0, 0, 0],
                        backgroundColor: ['#dc3545', '#ffc107', '#28a745']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { stepSize: 1 }
                        }
                    }
                }
            });

            const resCtx = document.getElementById('resolutionChart').getContext('2d');
            resolutionChart = new Chart(resCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Resolved', 'Active'],
                    datasets: [{
                        data: [0, 0],
                        backgroundColor: ['#28a745', '#dc3545']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }

        function updateCharts() {
            const timeRange = document.getElementById('vizTimeRange').value;
            const sosFilter = document.getElementById('vizSOSFilter').value;

            let filtered = [...allLogs];

            if (sosFilter === 'sos') {
                filtered = filtered.filter(log => log.sos);
            } else if (sosFilter === 'regular') {
                filtered = filtered.filter(log => !log.sos);
            }

            const now = new Date();
            if (timeRange !== 'all') {
                filtered = filtered.filter(log => {
                    const logDate = new Date(log.timestamp);
                    const diffDays = (now - logDate) / (1000 * 60 * 60 * 24);
                    
                    if (timeRange === 'today') return diffDays < 1;
                    if (timeRange === 'week') return diffDays < 7;
                    if (timeRange === 'month') return diffDays < 30;
                    return true;
                });
            }

            const timeData = {};
            filtered.forEach(log => {
                const date = new Date(log.timestamp).toLocaleDateString();
                timeData[date] = (timeData[date] || 0) + 1;
            });
            
            const sortedDates = Object.keys(timeData).sort((a, b) => new Date(a) - new Date(b));
            timeChart.data.labels = sortedDates;
            timeChart.data.datasets[0].data = sortedDates.map(date => timeData[date]);
            timeChart.update();

            const critical = filtered.filter(log => log.priority === 'CRITICAL').length;
            const urgent = filtered.filter(log => log.priority === 'URGENT').length;
            const standard = filtered.filter(log => log.priority === 'STANDARD').length;
            
            distributionChart.data.datasets[0].data = [critical, urgent, standard];
            distributionChart.update();

            const resolved = filtered.filter(log => log.resolved).length;
            const active = filtered.length - resolved;
            
            resolutionChart.data.datasets[0].data = [resolved, active];
            resolutionChart.update();
        }
    </script>
</body>
</html>
    """

    @app.route('/')
    def index():
        return render_template_string(TEMPLATE, department=department_name)

    return app

if __name__ == '__main__':
    import sys
    import threading
    
    departments = {
        'POLICE': 6001,
        'FIRE': 6002,
        'MEDICAL': 6003,
        'RESCUE': 6004,
        'UTILITIES': 6005
    }
    
    print("=" * 60)
    print("DEPARTMENT DASHBOARD SERVER")
    print("=" * 60)
    print("\nStarting department-specific dashboards...")
    print()
    
    def run_app(dept, port):
        app = create_department_app(dept, port)
        print(f"✓ {dept} Department: http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    threads = []
    for dept, port in departments.items():
        thread = threading.Thread(target=run_app, args=(dept, port))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print()
    print("=" * 60)
    print("All department dashboards are running!")
    print("Press Ctrl+C to stop all servers")
    print("=" * 60)
    
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n\nShutting down all servers...")