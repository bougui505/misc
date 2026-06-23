// Thermometer Dashboard Frontend Controller

// Configuration
const POLL_INTERVAL = 2000; // 2 seconds
let currentPeriod = '24h';
let chartInstance = null;

// DOM Elements
const currentTempEl = document.getElementById('current-temp');
const lastUpdateEl = document.getElementById('last-update');
const statusDotEl = document.getElementById('status-dot');
const statusTextEl = document.getElementById('status-text');
const tempCardEl = document.getElementById('temp-card');
const statMaxEl = document.getElementById('stat-max');
const statMinEl = document.getElementById('stat-min');
const statAvgEl = document.getElementById('stat-avg');
const timeframeButtons = document.querySelectorAll('.btn-timeframe');
const devicePathEl = document.getElementById('device-path');

// Helper to determine temperature range and update theme colors
function updateThemeForTemperature(temp) {
    if (temp === null || isNaN(temp)) return;
    
    let colorName;
    if (temp <= 15) {
        colorName = 'cold';
    } else if (temp <= 25) {
        colorName = 'normal';
    } else if (temp <= 32) {
        colorName = 'warm';
    } else {
        colorName = 'hot';
    }
    
    // Set dynamic CSS properties on document root
    document.documentElement.style.setProperty('--temp-active', `var(--temp-${colorName})`);
    document.documentElement.style.setProperty('--temp-active-glow', `var(--temp-${colorName}-glow)`);
}

// Format Unix timestamp to readable string depending on active period
function formatTimestamp(unixSecs, period) {
    const date = new Date(unixSecs * 1000);
    
    const pad = (n) => String(n).padStart(2, '0');
    
    const hours = pad(date.getHours());
    const mins = pad(date.getMinutes());
    const secs = pad(date.getSeconds());
    
    if (period === '1h') {
        return `${hours}:${mins}:${secs}`;
    } else if (period === '24h') {
        return `${hours}:${mins}`;
    } else { // '7d'
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const day = date.getDate();
        const month = months[date.getMonth()];
        return `${month} ${day} ${hours}:${mins}`;
    }
}

// Fetch and update the real-time current temperature
async function fetchCurrentTemp() {
    try {
        const response = await fetch('/api/current');
        if (!response.ok) throw new Error('API server returned error status');
        
        const data = await response.json();
        
        if (data.status === 'connected' && data.temperature !== null) {
            // Update value
            currentTempEl.textContent = data.temperature.toFixed(1);
            
            // Format last update time
            const lastTime = new Date(data.timestamp * 1000);
            lastUpdateEl.textContent = lastTime.toLocaleTimeString();
            
            // Update online state
            statusDotEl.className = 'status-dot online pulses';
            statusTextEl.textContent = 'Monitoring Live';
            
            // Adjust page colors according to current temperature
            updateThemeForTemperature(data.temperature);
        } else {
            // Sensor unplugged or serial read failure
            statusDotEl.className = 'status-dot offline';
            statusTextEl.textContent = data.error || 'Sensor Disconnected';
            
            // Keep previous values but show they are stale
            lastUpdateEl.textContent = data.timestamp ? `Stale (${new Date(data.timestamp * 1000).toLocaleTimeString()})` : 'Disconnected';
        }
        
    } catch (error) {
        console.error('Error fetching current temperature:', error);
        statusDotEl.className = 'status-dot offline';
        statusTextEl.textContent = 'Server Offline';
    }
}

// Calculate summary stats for the data points
function updateSummaryStats(data) {
    if (!data || data.length === 0) {
        statMaxEl.textContent = '--.-°C';
        statMinEl.textContent = '--.-°C';
        statAvgEl.textContent = '--.-°C';
        return;
    }
    
    const temps = data.map(d => d.temperature);
    const max = Math.max(...temps);
    const min = Math.min(...temps);
    const avg = temps.reduce((acc, v) => acc + v, 0) / temps.length;
    
    statMaxEl.textContent = `${max.toFixed(1)}°C`;
    statMinEl.textContent = `${min.toFixed(1)}°C`;
    statAvgEl.textContent = `${avg.toFixed(1)}°C`;
}

// Draw or update the Chart.js line graph
function drawChart(historyData) {
    const ctx = document.getElementById('tempChart').getContext('2d');
    
    const labels = historyData.map(d => formatTimestamp(d.timestamp, currentPeriod));
    const temps = historyData.map(d => d.temperature);
    
    // Create modern glowing area gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    const activeColor = getComputedStyle(document.documentElement).getPropertyValue('--temp-active').trim() || '#10b981';
    
    gradient.addColorStop(0, hexToRgbA(activeColor, 0.25));
    gradient.addColorStop(0.5, hexToRgbA(activeColor, 0.08));
    gradient.addColorStop(1, hexToRgbA(activeColor, 0.0));

    if (chartInstance) {
        // Update existing chart to prevent re-creation flicker
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = temps;
        chartInstance.data.datasets[0].borderColor = activeColor;
        chartInstance.data.datasets[0].backgroundColor = gradient;
        chartInstance.update();
    } else {
        // Create chart configuration
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Temperature',
                    data: temps,
                    borderColor: activeColor,
                    backgroundColor: gradient,
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointRadius: (context) => {
                        // Only draw point circles if there are few items on chart
                        const count = context.chart.data.datasets[0].data.length;
                        return count < 40 ? 3.5 : 0;
                    },
                    pointBackgroundColor: activeColor,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 1.5,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: activeColor,
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        titleColor: '#f3f4f6',
                        bodyColor: '#e5e7eb',
                        titleFont: { family: 'Outfit', weight: '600', size: 13 },
                        bodyFont: { family: 'Outfit', size: 13 },
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        padding: 10,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return `Temperature: ${context.parsed.y.toFixed(1)}°C`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.03)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 11 },
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 11 },
                            callback: function(value) {
                                return value.toFixed(1) + '°C';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Convert Hex colors to RGBA for standard canvas gradients
function hexToRgbA(hex, alpha) {
    let c;
    if(/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)){
        c= hex.substring(1).split('');
        if(c.length== 3){
            c= [c[0], c[0], c[1], c[1], c[2], c[2]];
        }
        c= '0x' + c.join('');
        return 'rgba('+[(c>>16)&255, (c>>8)&255, c&255].join(',')+','+alpha+')';
    }
    // Fallback to standard color representations if CSS variable resolves to named colors
    if (hex === 'var(--temp-cold)' || hex.includes('3b82f6')) return `rgba(59,130,246,${alpha})`;
    if (hex === 'var(--temp-normal)' || hex.includes('10b981')) return `rgba(16,185,129,${alpha})`;
    if (hex === 'var(--temp-warm)' || hex.includes('f59e0b')) return `rgba(245,158,11,${alpha})`;
    if (hex === 'var(--temp-hot)' || hex.includes('ef4444')) return `rgba(239,68,68,${alpha})`;
    return `rgba(16,185,129,${alpha})`;
}

// Fetch and load historical data
async function loadHistory(period) {
    try {
        const response = await fetch(`/api/history?period=${period}`);
        if (!response.ok) throw new Error('API returned history error status');
        
        const historyData = await response.json();
        
        updateSummaryStats(historyData);
        drawChart(historyData);
        
    } catch (error) {
        console.error('Error fetching historical temperature:', error);
    }
}

// Event Listeners for Period Button Switches
timeframeButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
        timeframeButtons.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        
        currentPeriod = e.target.getAttribute('data-period');
        loadHistory(currentPeriod);
    });
});

// Initialization
async function init() {
    // Initial fetch of status and temperature
    await fetchCurrentTemp();
    await loadHistory(currentPeriod);
    
    // Setup background loops
    setInterval(fetchCurrentTemp, POLL_INTERVAL);
    
    // Refresh history data less frequently to save server resources
    setInterval(() => loadHistory(currentPeriod), 15000);
}

// Start
document.addEventListener('DOMContentLoaded', init);
