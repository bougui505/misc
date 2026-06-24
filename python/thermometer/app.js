// Thermometer Dashboard Frontend Controller

// Configuration
const POLL_INTERVAL = 2000; // 2 seconds
let currentPeriod = '24h';
let chartInstance = null;
let activeChartPeriod = null;
let correlationChartInstance = null;
let outdoorForecast = null;

// DOM Elements
const currentTempEl = document.getElementById('current-temp');
const lastUpdateEl = document.getElementById('last-update');
const currentHumidityEl = document.getElementById('current-humidity');
const humidityLastUpdateEl = document.getElementById('humidity-last-update');
const currentFeelsLikeEl = document.getElementById('current-feels-like');
const statusDotEl = document.getElementById('status-dot');
const statusTextEl = document.getElementById('status-text');
const tempCardEl = document.getElementById('temp-card');
const statMaxEl = document.getElementById('stat-max');
const statMinEl = document.getElementById('stat-min');
const statAvgEl = document.getElementById('stat-avg');
const timeframeButtons = document.querySelectorAll('.btn-timeframe');
const devicePathEl = document.getElementById('device-path');

// Calculate feels-like (Australian Apparent Temperature formula)
function calculateFeelsLike(temp, humidity) {
    if (temp === null || humidity === null || isNaN(temp) || isNaN(humidity)) return null;
    // Water vapor pressure (e) in hPa using Magnus-Tetens approximation
    const e = (humidity / 100.0) * 6.105 * Math.exp((17.27 * temp) / (237.7 + temp));
    return temp + 0.33 * e - 4.00;
}

// Helper to determine temperature range and update theme colors
function updateThemeForTemperature(temp) {
    if (temp === null || isNaN(temp)) return;
    
    let colorName;
    if (temp <= 18) {
        colorName = 'cold';
    } else if (temp <= 21) {
        colorName = 'normal';
    } else if (temp <= 25) {
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

// Fetch and update the real-time current temperature and humidity
async function fetchCurrentTemp() {
    try {
        const response = await fetch('/api/current');
        if (!response.ok) throw new Error('API server returned error status');
        
        const data = await response.json();
        
        if (data.status === 'connected' && data.temperature !== null) {
            // Update values
            currentTempEl.textContent = data.temperature.toFixed(1);
            if (data.humidity !== null && data.humidity !== undefined) {
                currentHumidityEl.textContent = data.humidity.toFixed(1);
                
                // Calculate and update Feels Like
                const feelsLike = calculateFeelsLike(data.temperature, data.humidity);
                if (feelsLike !== null) {
                    currentFeelsLikeEl.textContent = feelsLike.toFixed(1);
                } else {
                    currentFeelsLikeEl.textContent = '--.-';
                }
            } else {
                currentHumidityEl.textContent = '--.-';
                currentFeelsLikeEl.textContent = '--.-';
            }
            
            // Format last update time
            const lastTime = new Date(data.timestamp * 1000);
            const pad = (n) => String(n).padStart(2, '0');
            const timeStr = `${pad(lastTime.getHours())}:${pad(lastTime.getMinutes())}:${pad(lastTime.getSeconds())}`;
            lastUpdateEl.textContent = timeStr;
            humidityLastUpdateEl.textContent = timeStr;
            
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
            let staleTime = 'Disconnected';
            if (data.timestamp) {
                const staleDate = new Date(data.timestamp * 1000);
                const pad = (n) => String(n).padStart(2, '0');
                const timeStr = `${pad(staleDate.getHours())}:${pad(staleDate.getMinutes())}:${pad(staleDate.getSeconds())}`;
                staleTime = `Stale (${timeStr})`;
            }
            lastUpdateEl.textContent = staleTime;
            humidityLastUpdateEl.textContent = staleTime;
            currentFeelsLikeEl.textContent = '--.-';
        }
        
    } catch (error) {
        console.error('Error fetching current temperature and humidity:', error);
        statusDotEl.className = 'status-dot offline';
        statusTextEl.textContent = 'Server Offline';
    }
}

// Calculate summary stats for the data points (Temperature stats)
function updateSummaryStats(data) {
    if (!data || data.length === 0) {
        statMaxEl.textContent = '--.-°C';
        statMinEl.textContent = '--.-°C';
        statAvgEl.textContent = '--.-°C';
        return;
    }
    
    // For comparison view, filter stats to only represent the last 24 hours (Today)
    const refTime = data[data.length - 1].timestamp;
    const statsData = data.filter(d => refTime - d.timestamp < 86400);
    
    const temps = statsData.map(d => d.temperature);
    if (temps.length === 0) return;
    
    const max = Math.max(...temps);
    const min = Math.min(...temps);
    const avg = temps.reduce((acc, v) => acc + v, 0) / temps.length;
    
    statMaxEl.textContent = `${max.toFixed(1)}°C`;
    statMinEl.textContent = `${min.toFixed(1)}°C`;
    statAvgEl.textContent = `${avg.toFixed(1)}°C`;
}

// Draw or update the Chart.js line graph
function drawChart(historyData) {
    // If the timeframe has switched, destroy the old chart so we can reconstruct datasets cleanly
    if (chartInstance && activeChartPeriod !== currentPeriod) {
        chartInstance.destroy();
        chartInstance = null;
    }
    
    const ctx = document.getElementById('tempChart').getContext('2d');
    
    let labels, dataset1, dataset2, dataset4, dataset4Colors;
    let label1, label2;
    let color1, color2;
    
    const activeColor = getComputedStyle(document.documentElement).getPropertyValue('--temp-active').trim() || '#10b981';
    const feelsColor = '#a855f7';
    
    if (currentPeriod === 'compare' || currentPeriod === 'anomaly') {
        const now = historyData.length > 0 ? historyData[historyData.length - 1].timestamp : Math.floor(Date.now() / 1000);
        const numBuckets = 288; // 24 hours / 5 min
        const bucketSize = 300; // 5 min in seconds

        // Initialize arrays
        const todayTemps = new Array(numBuckets).fill(null);
        const yesterdayTemps = new Array(numBuckets).fill(null);
        labels = new Array(numBuckets);

        // Populate labels (moving backwards from now)
        for (let i = 0; i < numBuckets; i++) {
            const ts = now - i * bucketSize;
            labels[numBuckets - 1 - i] = formatTimestamp(ts, '24h');
        }

        // Populate data
        historyData.forEach(d => {
            const age = now - d.timestamp;
            if (age >= 0 && age < 86400) {
                // Today
                const bucketIdx = Math.floor(age / bucketSize);
                if (bucketIdx >= 0 && bucketIdx < numBuckets) {
                    todayTemps[numBuckets - 1 - bucketIdx] = d.temperature;
                }
            } else if (age >= 86400 && age < 2 * 86400) {
                // Yesterday
                const bucketIdx = Math.floor((age - 86400) / bucketSize);
                if (bucketIdx >= 0 && bucketIdx < numBuckets) {
                    yesterdayTemps[numBuckets - 1 - bucketIdx] = d.temperature;
                }
            }
        });
        
        if (currentPeriod === 'anomaly') {
            // Calculate anomaly values (Today - Yesterday)
            const anomalies = new Array(numBuckets).fill(null);
            for (let i = 0; i < numBuckets; i++) {
                if (todayTemps[i] !== null && yesterdayTemps[i] !== null) {
                    anomalies[i] = parseFloat((todayTemps[i] - yesterdayTemps[i]).toFixed(2));
                }
            }
            
            dataset1 = anomalies;
            dataset2 = null;
            label1 = "Today's Temp Deviation from Yesterday";
        } else {
            // 'compare' overlaid lines
            dataset1 = todayTemps;
            dataset2 = yesterdayTemps;
            label1 = "Today's Temperature";
            label2 = "Yesterday's Temperature";
            color1 = activeColor;
            color2 = '#60a5fa'; // Sleek sky blue for yesterday
        }
    } else if (currentPeriod === 'ventilation' || currentPeriod === 'ventilation_deviation') {
        const refNow = historyData.length > 0 ? historyData[historyData.length - 1].timestamp : Math.floor(Date.now() / 1000);
        // Round refNow to the nearest hour
        const nowHourTS = Math.round(refNow / 3600) * 3600;
        
        const numPastHours = 24;
        const numFutureHours = 24;
        const totalHours = numPastHours + numFutureHours + 1; // 49 hours total (24 past, now, 24 future)
        
        labels = new Array(totalHours);
        const actualIndoor = new Array(totalHours).fill(null);
        const predictedIndoor = new Array(totalHours).fill(null);
        const outdoorDataPoints = new Array(totalHours).fill(null);
        
        const getIsoHourString = (dateObj) => {
            const yyyy = dateObj.getFullYear();
            const mm = String(dateObj.getMonth() + 1).padStart(2, '0');
            const dd = String(dateObj.getDate()).padStart(2, '0');
            const hh = String(dateObj.getHours()).padStart(2, '0');
            return `${yyyy}-${mm}-${dd}T${hh}:00`;
        };
        
        // Loop over the 49-hour span to populate past actual values and outdoor forecast values
        for (let offset = -numPastHours; offset <= numFutureHours; offset++) {
            const ts = nowHourTS + offset * 3600;
            const dateObj = new Date(ts * 1000);
            const idx = offset + numPastHours;
            
            labels[idx] = formatTimestamp(ts, '7d');
            
            // 1. Get Outdoor Temperature from Open-Meteo forecast (available for both past and future)
            if (outdoorForecast && outdoorForecast.hourly) {
                const iso = getIsoHourString(dateObj);
                const fIdx = outdoorForecast.hourly.time.indexOf(iso);
                if (fIdx !== -1) {
                    outdoorDataPoints[idx] = outdoorForecast.hourly.temperature_2m[fIdx];
                }
            }
            
            // 2. Average actual indoor readings for past hours
            if (offset <= 0) {
                const hourReadings = historyData.filter(d => Math.abs(d.timestamp - ts) < 1800);
                if (hourReadings.length > 0) {
                    actualIndoor[idx] = parseFloat((hourReadings.reduce((sum, r) => sum + r.temperature, 0) / hourReadings.length).toFixed(2));
                } else if (offset === 0 && historyData.length > 0) {
                    // Fallback for current hour
                    actualIndoor[idx] = historyData[historyData.length - 1].temperature;
                }
            }
        }
        
        // 3. Connect predicted line to the last actual reading at index 24 (Now)
        predictedIndoor[numPastHours] = actualIndoor[numPastHours];
        
        // 4. Recursive thermal prediction for future hours (windows assumed closed)
        for (let offset = 1; offset <= numFutureHours; offset++) {
            const idx = offset + numPastHours;
            const prevIdx = idx - 1;
            const prevIndoor = (offset === 1) ? actualIndoor[numPastHours] : predictedIndoor[prevIdx];
            const outTemp = outdoorDataPoints[idx];
            
            if (prevIndoor !== null && outTemp !== null) {
                // T_in = T_prev + k*(T_out - T_prev) + internal_heating
                // k = 0.05 (thermal transfer coefficient), internal_heating = 0.03°C/hour
                predictedIndoor[idx] = parseFloat((prevIndoor + 0.05 * (outTemp - prevIndoor) + 0.03).toFixed(2));
            }
        }
          if (currentPeriod === 'ventilation_deviation') {
            const deviations = new Array(totalHours).fill(null);
            for (let idx = 0; idx < totalHours; idx++) {
                const inTemp = idx <= numPastHours ? actualIndoor[idx] : predictedIndoor[idx];
                const outTemp = outdoorDataPoints[idx];
                if (inTemp !== null && outTemp !== null) {
                    deviations[idx] = parseFloat((outTemp - inTemp).toFixed(2));
                }
            }
            dataset1 = deviations;
            dataset2 = null;
            label1 = "Ventilation Temp Deviation (Outdoor - Indoor)";
        } else {
            dataset1 = actualIndoor;
            dataset2 = predictedIndoor;
            dataset3 = outdoorDataPoints;
            label1 = 'Indoor Temp (Actual)';
            label2 = 'Indoor Temp (Predicted)';
            label3 = 'Outdoor Temp (Forecast)';
            color1 = '#06b6d4'; // Cool Teal for Indoor Actual
            color2 = '#06b6d4'; // Cool Teal for Indoor Predicted
            color3 = '#f59e0b'; // Amber for Outdoor Forecast
            
            // Calculate comfort window actions to highlight on the plot background
            const COMFORT_MIN = 19.0;
            const COMFORT_MAX = 21.0;
            dataset4 = new Array(totalHours).fill(0);
            dataset4Colors = new Array(totalHours).fill('rgba(0,0,0,0)');
            for (let idx = 0; idx < totalHours; idx++) {
                const inTemp = idx <= numPastHours ? actualIndoor[idx] : predictedIndoor[idx];
                const outTemp = outdoorDataPoints[idx];
                if (inTemp !== null && outTemp !== null) {
                    let isOpen = false;
                    if (inTemp < COMFORT_MIN) {
                        isOpen = outTemp > inTemp;
                    } else if (inTemp > COMFORT_MAX) {
                        isOpen = outTemp < inTemp;
                    } else {
                        isOpen = outTemp >= COMFORT_MIN && outTemp <= COMFORT_MAX;
                    }
                    dataset4[idx] = 1;
                    dataset4Colors[idx] = isOpen ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.04)';
                }
            }
        }
    } else {
        labels = historyData.map(d => formatTimestamp(d.timestamp, currentPeriod));
        dataset1 = historyData.map(d => d.temperature);
        dataset2 = historyData.map(d => {
            const feels = calculateFeelsLike(d.temperature, d.humidity);
            return feels !== null ? parseFloat(feels.toFixed(2)) : d.temperature;
        });
        label1 = 'Temperature';
        label2 = 'Feels Like';
        color1 = activeColor;
        color2 = feelsColor;
    }
    
    // Create modern glowing area gradients
    const tempGradient = ctx.createLinearGradient(0, 0, 0, 300);
    tempGradient.addColorStop(0, hexToRgbA(color1, 0.25));
    tempGradient.addColorStop(0.5, hexToRgbA(color1, 0.08));
    tempGradient.addColorStop(1, hexToRgbA(color1, 0.0));

    const feelsGradient = ctx.createLinearGradient(0, 0, 0, 300);
    feelsGradient.addColorStop(0, hexToRgbA(color2, 0.25));
    feelsGradient.addColorStop(0.5, hexToRgbA(color2, 0.08));
    feelsGradient.addColorStop(1, hexToRgbA(color2, 0.0));

    // Prepare datasets array dynamically
    const chartDatasets = [];
    if (currentPeriod === 'anomaly' || currentPeriod === 'ventilation_deviation') {
        const borderColors = dataset1.map(v => v !== null ? (v >= 0 ? '#ef4444' : '#3b82f6') : 'transparent');
        const backgroundColors = dataset1.map(v => v !== null ? (v >= 0 ? 'rgba(239, 68, 68, 0.35)' : 'rgba(59, 130, 246, 0.35)') : 'rgba(0,0,0,0)');
        
        chartDatasets.push({
            label: label1,
            data: dataset1,
            borderColor: borderColors,
            backgroundColor: backgroundColors,
            borderWidth: 1.5,
            borderRadius: 4,
            yAxisID: 'y'
        });
    } else if (currentPeriod === 'ventilation') {
        // Shaded background bar chart to highlight action recommendation regions
        chartDatasets.push({
            type: 'bar',
            label: 'Action Suggestion',
            data: dataset4,
            backgroundColor: dataset4Colors,
            borderColor: 'transparent',
            borderWidth: 0,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
            yAxisID: 'y2',
            order: 2
        });
        
        chartDatasets.push({
            label: label1,
            data: dataset1,
            borderColor: color1,
            borderWidth: 2.5,
            fill: false,
            tension: 0.3,
            yAxisID: 'y',
            pointRadius: 0,
            spanGaps: true,
            order: 1
        });
        chartDatasets.push({
            label: label2,
            data: dataset2,
            borderColor: color2,
            borderWidth: 2.5,
            borderDash: [5, 5],
            fill: false,
            tension: 0.3,
            yAxisID: 'y',
            pointRadius: 0,
            spanGaps: true,
            order: 1
        });
        chartDatasets.push({
            label: label3,
            data: dataset3,
            borderColor: color3,
            borderWidth: 2.5,
            fill: false,
            tension: 0.3,
            yAxisID: 'y',
            pointRadius: 0,
            spanGaps: true,
            order: 1
        });
    } else {
        chartDatasets.push({
            label: label1,
            data: dataset1,
            borderColor: color1,
            backgroundColor: tempGradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            yAxisID: 'y',
            pointRadius: (context) => {
                const count = context.chart.data.datasets[0].data.length;
                return count < 40 ? 3.5 : 0;
            },
            pointBackgroundColor: color1,
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.5,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: color1,
            pointHoverBorderColor: '#ffffff',
            pointHoverBorderWidth: 2,
            spanGaps: true
        });
        
        chartDatasets.push({
            label: label2,
            data: dataset2,
            borderColor: color2,
            backgroundColor: feelsGradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            yAxisID: 'y',
            pointRadius: (context) => {
                const count = context.chart.data.datasets[1].data.length;
                return count < 40 ? 3.5 : 0;
            },
            pointBackgroundColor: color2,
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.5,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: color2,
            pointHoverBorderColor: '#ffffff',
            pointHoverBorderWidth: 2,
            spanGaps: true
        });
    }

    if (chartInstance) {
        // Update the existing chart smoothly in-place without blinking
        chartInstance.data.labels = labels;
        
        if (currentPeriod === 'anomaly' || currentPeriod === 'ventilation_deviation') {
            const borderColors = dataset1.map(v => v !== null ? (v >= 0 ? '#ef4444' : '#3b82f6') : 'transparent');
            const backgroundColors = dataset1.map(v => v !== null ? (v >= 0 ? 'rgba(239, 68, 68, 0.35)' : 'rgba(59, 130, 246, 0.35)') : 'rgba(0,0,0,0)');
            
            chartInstance.data.datasets[0].data = dataset1;
            chartInstance.data.datasets[0].label = label1;
            chartInstance.data.datasets[0].borderColor = borderColors;
            chartInstance.data.datasets[0].backgroundColor = backgroundColors;
        } else if (currentPeriod === 'ventilation') {
            chartInstance.data.datasets[0].data = dataset4;
            chartInstance.data.datasets[0].backgroundColor = dataset4Colors;
            
            chartInstance.data.datasets[1].data = dataset1;
            chartInstance.data.datasets[1].label = label1;
            
            chartInstance.data.datasets[2].data = dataset2;
            chartInstance.data.datasets[2].label = label2;
            
            chartInstance.data.datasets[3].data = dataset3;
            chartInstance.data.datasets[3].label = label3;
        } else {
            chartInstance.data.datasets[0].data = dataset1;
            chartInstance.data.datasets[0].label = label1;
            chartInstance.data.datasets[0].borderColor = color1;
            chartInstance.data.datasets[0].backgroundColor = tempGradient;
            chartInstance.data.datasets[0].pointBackgroundColor = color1;
            chartInstance.data.datasets[0].pointHoverBackgroundColor = color1;
            
            chartInstance.data.datasets[1].data = dataset2;
            chartInstance.data.datasets[1].label = label2;
            chartInstance.data.datasets[1].borderColor = color2;
            chartInstance.data.datasets[1].backgroundColor = feelsGradient;
            chartInstance.data.datasets[1].pointBackgroundColor = color2;
            chartInstance.data.datasets[1].pointHoverBackgroundColor = color2;
        }
        
        chartInstance.update('none'); // Update without animation during ticks to prevent blinking
    } else {
        activeChartPeriod = currentPeriod;
        // Create chart configuration
        chartInstance = new Chart(ctx, {
            type: (currentPeriod === 'anomaly' || currentPeriod === 'ventilation_deviation') ? 'bar' : 'line',
            data: {
                labels: labels,
                datasets: chartDatasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 12 },
                            filter: function(item, chartData) {
                                return item.text !== 'Action Suggestion';
                            }
                        }
                    },
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
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                const val = context.parsed.y;
                                 if (currentPeriod === 'anomaly' || currentPeriod === 'ventilation_deviation') {
                                     return `Deviation: ${val >= 0 ? '+' : ''}${val.toFixed(2)}°C`;
                                 }
                                const label = context.dataset.label || '';
                                return `${label}: ${val !== null && val !== undefined ? val.toFixed(1) : '--.-'}°C`;
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
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 11 },
                            callback: function(value) {
                                 if (currentPeriod === 'anomaly' || currentPeriod === 'ventilation_deviation') {
                                     return (value >= 0 ? '+' : '') + value.toFixed(1) + '°C';
                                 }
                                return value.toFixed(1) + '°C';
                            }
                        }
                    },
                    y2: {
                        type: 'linear',
                        display: false,
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }
}

// Convert Hex colors to RGBA for standard canvas gradients
function hexToRgbA(hex, alpha) {
    if (!hex || typeof hex !== 'string') return `rgba(16,185,129,${alpha})`;
    
    // If it's a hex color
    if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
        let c = hex.substring(1).split('');
        if (c.length == 3) {
            c = [c[0], c[0], c[1], c[1], c[2], c[2]];
        }
        c = '0x' + c.join('');
        return 'rgba(' + [(c >> 16) & 255, (c >> 8) & 255, c & 255].join(',') + ',' + alpha + ')';
    }
    
    // Fallback to standard color representations if CSS variable resolves to named colors
    if (hex.includes('cold') || hex.includes('3b82f6')) return `rgba(59,130,246,${alpha})`;
    if (hex.includes('normal') || hex.includes('10b981')) return `rgba(16,185,129,${alpha})`;
    if (hex.includes('warm') || hex.includes('eab308')) return `rgba(234,179,8,${alpha})`;
    if (hex.includes('hot') || hex.includes('ef4444')) return `rgba(239,68,68,${alpha})`;
    return `rgba(16,185,129,${alpha})`;
}

// Fetch and load historical data
async function loadHistory(period) {
    try {
        let fetchPeriod = period;
        let outdoorData = null;
        
        if (period === 'ventilation' || period === 'ventilation_deviation') {
            fetchPeriod = '24h';
            // Fetch outdoor forecast from Open-Meteo for Paris coordinates
            try {
                const lat = 48.8566;
                const lon = 2.3522;
                const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m&timezone=auto`;
                const response = await fetch(url);
                if (response.ok) {
                    outdoorData = await response.json();
                    outdoorForecast = outdoorData;
                }
            } catch (err) {
                console.error("Error fetching Open-Meteo outdoor forecast:", err);
            }
        }
        
        const response = await fetch(`/api/history?period=${fetchPeriod}`);
        if (!response.ok) throw new Error('API returned history error status');
        
        const historyData = await response.json();
        
        updateSummaryStats(historyData);
        
        if (period === 'ventilation' || period === 'ventilation_deviation') {
            // Match each indoor reading with the closest hourly outdoor forecast reading
            if (outdoorData && outdoorData.hourly) {
                const hourlyTimes = outdoorData.hourly.time;
                const hourlyTemps = outdoorData.hourly.temperature_2m;
                
                historyData.forEach(d => {
                    const dateObj = new Date(d.timestamp * 1000);
                    const yyyy = dateObj.getFullYear();
                    const mm = String(dateObj.getMonth() + 1).padStart(2, '0');
                    const dd = String(dateObj.getDate()).padStart(2, '0');
                    const hh = String(dateObj.getHours()).padStart(2, '0');
                    const isoStr = `${yyyy}-${mm}-${dd}T${hh}:00`;
                    
                    const idx = hourlyTimes.indexOf(isoStr);
                    d.outdoorTemperature = idx !== -1 ? hourlyTemps[idx] : null;
                });
            }
        }
        
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

// Calculate and display historical insights
function calculateClimateInsights(history7d) {
    if (!history7d || history7d.length === 0) return;
    
    // Group by local date string
    const dailyData = {};
    const hourlyData = {};
    const scatterData = [];
    
    // Pearson correlation variables
    let n = 0;
    let sumX = 0, sumY = 0;
    let sumX2 = 0, sumY2 = 0, sumXY = 0;
    
    history7d.forEach(d => {
        const dateObj = new Date(d.timestamp * 1000);
        const dayStr = dateObj.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        const hourStr = String(dateObj.getHours()).padStart(2, '0');
        
        // Group by day
        if (!dailyData[dayStr]) {
            dailyData[dayStr] = { temps: [], hums: [] };
        }
        dailyData[dayStr].temps.push(d.temperature);
        if (d.humidity !== null && d.humidity !== undefined) {
            dailyData[dayStr].hums.push(d.humidity);
            scatterData.push({ x: d.temperature, y: d.humidity });
            
            // Correlation values
            n++;
            const x = d.temperature;
            const y = d.humidity;
            sumX += x;
            sumY += y;
            sumX2 += x * x;
            sumY2 += y * y;
            sumXY += x * y;
        }
        
        // Group by hour
        if (!hourlyData[hourStr]) {
            hourlyData[hourStr] = { temps: [] };
        }
        hourlyData[hourStr].temps.push(d.temperature);
    });
    
    // 1. Render Daily Summaries Table
    const tbody = document.getElementById('analytics-table-body');
    if (tbody) {
        tbody.innerHTML = '';
        
        // Get sorted list of days in reverse chronological order
        const days = Object.keys(dailyData).reverse().slice(0, 5); // last 5 days
        days.forEach(day => {
            const temps = dailyData[day].temps;
            const hums = dailyData[day].hums;
            
            const minT = Math.min(...temps);
            const maxT = Math.max(...temps);
            const avgT = temps.reduce((a, b) => a + b, 0) / temps.length;
            const avgH = hums.length > 0 ? (hums.reduce((a, b) => a + b, 0) / hums.length) : null;
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${day}</strong></td>
                <td><span style="color:var(--temp-cold);font-weight:500;">${minT.toFixed(1)}°C</span></td>
                <td><span style="color:var(--temp-hot);font-weight:500;">${maxT.toFixed(1)}°C</span></td>
                <td><strong>${avgT.toFixed(1)}°C</strong></td>
                <td>${avgH !== null ? avgH.toFixed(1) + '%' : '—'}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // 1.5. Render Ventilation Schedule (Next 8 Hours)
    const scheduleBody = document.getElementById('ventilation-schedule-body');
    if (scheduleBody && outdoorForecast && outdoorForecast.hourly) {
        scheduleBody.innerHTML = '';
        const nowTS = Math.floor(Date.now() / 1000);
        const nowHourTS = Math.round(nowTS / 3600) * 3600;
        
        let lastIndoor = history7d[history7d.length - 1].temperature;
        
        const getIsoHourString = (d) => {
            const yyyy = d.getFullYear();
            const mm = String(d.getMonth() + 1).padStart(2, '0');
            const dd = String(d.getDate()).padStart(2, '0');
            const hh = String(d.getHours()).padStart(2, '0');
            return `${yyyy}-${mm}-${dd}T${hh}:00`;
        };
        
        // Loop for the next 8 hours (starting at the current hour)
        let currentPred = lastIndoor;
        for (let h = 0; h < 8; h++) {
            const ts = nowHourTS + h * 3600;
            const dateObj = new Date(ts * 1000);
            const pad = (n) => String(n).padStart(2, '0');
            const timeStr = `${pad(dateObj.getHours())}:${pad(dateObj.getMinutes())}`;
            
            let outTemp = null;
            const iso = getIsoHourString(dateObj);
            const fIdx = outdoorForecast.hourly.time.indexOf(iso);
            if (fIdx !== -1) {
                outTemp = outdoorForecast.hourly.temperature_2m[fIdx];
            }
            
            if (outTemp !== null) {
                if (h > 0) {
                    // Predict step only for future hours
                    currentPred = currentPred + 0.05 * (outTemp - currentPred) + 0.03;
                }
                
                const delta = outTemp - currentPred;
                const pct = Math.min((Math.abs(delta) / 10) * 50, 50); // Scale 10°C to 50% max width
                const barSide = delta < 0 ? `right: 50%; width: ${pct}%;` : `left: 50%; width: ${pct}%;`;
                const barClass = delta < 0 ? 'cool' : 'warm';
                const sign = delta > 0 ? '+' : '';
                
                const deltaHtml = `
                    <div class="delta-bar-container">
                        <span class="delta-bar-label">${sign}${delta.toFixed(1)}°C</span>
                        <div class="delta-bar-track">
                            <div class="delta-bar-fill ${barClass}" style="${barSide}"></div>
                            <div class="delta-bar-center"></div>
                        </div>
                    </div>
                `;
                
                const COMFORT_MIN = 19.0;
                const COMFORT_MAX = 21.0;
                let isOpen = false;
                
                if (currentPred < COMFORT_MIN) {
                    // Indoor is too cold: open only if outdoor is warmer than indoor to help heat the room
                    isOpen = outTemp > currentPred;
                } else if (currentPred > COMFORT_MAX) {
                    // Indoor is too hot: open only if outdoor is cooler than indoor to help cool the room
                    isOpen = outTemp < currentPred;
                } else {
                    // Indoor is in comfort range: open only if outdoor air is also comfortable to maintain state
                    isOpen = outTemp >= COMFORT_MIN && outTemp <= COMFORT_MAX;
                }
                
                const verdict = isOpen 
                    ? '<span style="color:var(--success);font-weight:bold;">🔓 OPEN</span>' 
                    : '<span style="color:var(--danger);font-weight:bold;">🔒 CLOSE</span>';
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${timeStr}</strong></td>
                    <td>${currentPred.toFixed(1)}°C</td>
                    <td>${outTemp.toFixed(1)}°C</td>
                    <td>${deltaHtml}</td>
                    <td>${verdict}</td>
                `;
                scheduleBody.appendChild(tr);
            }
        }
    }
    
    // 2. Determine Best Ventilation Time (coolest hour)
    let coolestHour = '07';
    let minHourlyAvg = Infinity;
    Object.keys(hourlyData).forEach(hour => {
        const temps = hourlyData[hour].temps;
        const avg = temps.reduce((a, b) => a + b, 0) / temps.length;
        if (avg < minHourlyAvg) {
            minHourlyAvg = avg;
            coolestHour = hour;
        }
    });
    
    const ventilationEl = document.getElementById('insight-ventilation');
    if (ventilationEl) {
        const startHour = parseInt(coolestHour);
        const endHour = (startHour + 2) % 24;
        const pad = (h) => String(h).padStart(2, '0');
        ventilationEl.innerHTML = `Open windows between <strong>${pad(startHour)}:00 and ${pad(endHour)}:00</strong> when indoor temperature drops to its daily low (avg <strong>${minHourlyAvg.toFixed(1)}°C</strong>).`;
    }
    
    // 3. Calculate Correlation Coefficient & Linear Regression Line
    let corrCoef = 0;
    let regressionLine = [];
    const denom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    if (denom !== 0) {
        corrCoef = (n * sumXY - sumX * sumY) / denom;
    }
    
    if (n > 1 && denom !== 0) {
        const meanX = sumX / n;
        const meanY = sumY / n;
        const num = n * sumXY - sumX * sumY;
        const den = n * sumX2 - sumX * sumX;
        
        if (den !== 0) {
            const slope = num / den;
            const intercept = meanY - slope * meanX;
            
            const tempsOnly = scatterData.map(p => p.x);
            const minX = Math.min(...tempsOnly);
            const maxX = Math.max(...tempsOnly);
            
            regressionLine = [
                { x: minX, y: slope * minX + intercept },
                { x: maxX, y: slope * maxX + intercept }
            ];
        }
    }
    
    const correlationEl = document.getElementById('insight-correlation');
    if (correlationEl) {
        let desc = '';
        if (corrCoef > 0.4) {
            desc = `<strong>Strong positive (${corrCoef.toFixed(2)})</strong>: Humidity rises as it warms up. Suggests stagnant air; run an exhaust fan or ventilate when heating.`;
        } else if (corrCoef < -0.4) {
            desc = `<strong>Standard negative (${corrCoef.toFixed(2)})</strong>: Relative humidity behaves normally (drops as air warms). Good natural thermodynamics.`;
        } else {
            desc = `<strong>Neutral (${corrCoef.toFixed(2)})</strong>: No direct correlation between heat and humidity in this room.`;
        }
        correlationEl.innerHTML = desc;
    }
    
    // 4. Render Correlation Scatter Plot
    const corrCanvas = document.getElementById('correlationChart');
    if (corrCanvas && scatterData.length > 0) {
        if (correlationChartInstance) {
            correlationChartInstance.destroy();
            correlationChartInstance = null;
        }
        
        const corrCtx = corrCanvas.getContext('2d');
        correlationChartInstance = new Chart(corrCtx, {
            data: {
                datasets: [
                    {
                        type: 'scatter',
                        label: 'Readings',
                        data: scatterData,
                        backgroundColor: 'rgba(96, 165, 250, 0.45)',
                        borderColor: 'rgba(96, 165, 250, 0.75)',
                        borderWidth: 1,
                        pointRadius: 2,
                        pointHoverRadius: 4
                    },
                    {
                        type: 'line',
                        label: 'Trendline',
                        data: regressionLine,
                        borderColor: '#ef4444',
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0,
                        tension: 0,
                        showLine: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Temp: ${context.parsed.x.toFixed(1)}°C, Hum: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Temperature (°C)',
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 9, weight: '500' }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.02)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 9 }
                        }
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Humidity (%)',
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 9, weight: '500' }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.02)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 9 }
                        }
                    }
                }
            }
        });
    }
}

// Initialization
async function init() {
    // Initial fetch of status and temperature
    await fetchCurrentTemp();
    
    // Fetch outdoor forecast immediately on startup for Paris to ensure it's available for insights
    try {
        const lat = 48.8566;
        const lon = 2.3522;
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m&timezone=auto`;
        const response = await fetch(url);
        if (response.ok) {
            outdoorForecast = await response.json();
        }
    } catch (err) {
        console.error("Error fetching outdoor forecast on init:", err);
    }
    
    await loadHistory(currentPeriod);
    
    // Fetch 7d history once to populate the insights table
    try {
        const res7d = await fetch('/api/history?period=7d');
        if (res7d.ok) {
            const data7d = await res7d.json();
            calculateClimateInsights(data7d);
        }
    } catch (e) {
        console.error("Error fetching 7d history for insights:", e);
    }
    
    // Setup background loops
    setInterval(fetchCurrentTemp, POLL_INTERVAL);
    
    // Refresh history data less frequently to save server resources
    setInterval(() => loadHistory(currentPeriod), 15000);
}

// Start
document.addEventListener('DOMContentLoaded', init);
