// Thermometer Dashboard Frontend Controller

// Configuration
const POLL_INTERVAL = 2000; // 2 seconds
const LATITUDE = 48.8285;  // Rue Sarrette, 75014 Paris
const LONGITUDE = 2.3315; // Rue Sarrette, 75014 Paris
let currentPeriod = '24h';
let chartInstance = null;
let activeChartPeriod = null;
let humidityChartInstance = null;
let activeHumidityChartPeriod = null;
let correlationChartInstance = null;
let outdoorForecast = null;
let forecastErrorsList = [];

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

// Calculate solar bias using month-based seasonal scaling and cloud cover
function getSolarParameters(date, cloudCover) {
    if (!date) return 0.0;
    const month = date.getMonth(); // 0-indexed (0: Jan, 1: Feb, 2: Mar, etc.)
    const hourFraction = date.getHours() + date.getMinutes() / 60.0;
    
    let startHour, endHour, maxBias;
    
    // Summer: May-Aug (months 4, 5, 6, 7)
    if (month >= 4 && month <= 7) {
        startHour = 14.0; // 14:00
        endHour = 20.0;   // 20:00
        maxBias = 4.0;
    } 
    // Spring/Autumn: Mar-Apr (months 2, 3) & Sep-Oct (months 8, 9)
    else if (month === 2 || month === 3 || month === 8 || month === 9) {
        startHour = 13.8333; // 13:50 (13 + 50/60)
        endHour = 19.0;      // 19:00
        maxBias = 4.5;
    } 
    // Winter: Nov-Feb (months 10, 11, 0, 1)
    else {
        startHour = 14.0; // 14:00
        endHour = 17.25;  // 17:15 (17 + 15/60)
        maxBias = 2.5;
    }
    
    if (hourFraction >= startHour && hourFraction <= endHour) {
        const cloudFactor = 1.0 - (cloudCover || 0) / 100.0;
        return maxBias * cloudFactor;
    }
    
    return 0.0;
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

// Custom plugin to draw a vertical line representing "Now" on forecast/deviation charts
const verticalLinePlugin = {
    id: 'verticalLine',
    afterDraw: (chart) => {
        if (currentPeriod === 'forecast' || currentPeriod === 'forecast_deviation') {
            const ctx = chart.ctx;
            const xAxis = chart.scales.x;
            const yAxis = chart.scales.y;
            
            // The current time is at index 24 (numPastHours)
            const index = 24;
            const meta = chart.getDatasetMeta(0);
            if (!meta || !meta.data || !meta.data[index]) return;
            const x = meta.data[index].x;
            const yTop = yAxis.top;
            const yBottom = yAxis.bottom;
            
            ctx.save();
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.75)'; // Semi-transparent Red
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]); // Dashed line
            ctx.moveTo(x, yTop);
            ctx.lineTo(x, yBottom);
            ctx.stroke();
            
            // Draw a red "NOW" badge at the top of the line
            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 9px Outfit, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            const text = 'NOW';
            const textWidth = ctx.measureText(text).width;
            const padX = 5;
            const padY = 3;
            const badgeW = textWidth + padX * 2;
            const badgeH = 13;
            const badgeX = x - badgeW / 2;
            const badgeY = yTop - badgeH - 3;
            
            // Background pill
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 3);
            } else {
                ctx.rect(badgeX, badgeY, badgeW, badgeH);
            }
            ctx.fillStyle = '#ef4444';
            ctx.fill();
            
            // Text inside pill
            ctx.fillStyle = '#ffffff';
            ctx.fillText(text, x, badgeY + badgeH / 2 + 0.5);
            
            ctx.restore();
        }
    }
};

// Custom plugin to draw highlights and labels on the indoor temperature extrema (max and min) in forecast mode
const forecastExtremaPlugin = {
    id: 'forecastExtrema',
    afterDraw: (chart) => {
        if (currentPeriod !== 'forecast') return;
        const ctx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        if (!yAxis || !xAxis) return;
        
        // Find datasets representing indoor temperatures
        const dsActual = chart.data.datasets.find(ds => ds.label === 'Indoor Temp (Actual)');
        const dsPredicted = chart.data.datasets.find(ds => ds.label === 'Indoor Temp (Predicted)');
        if (!dsActual || !dsPredicted) return;
        
        // Construct combined internal temperatures
        const dataLength = chart.data.labels.length;
        let maxVal = -Infinity, minVal = Infinity;
        let maxIdx = -1, minIdx = -1;
        
        for (let i = 0; i < dataLength; i++) {
            const act = dsActual.data[i];
            const pred = dsPredicted.data[i];
            const val = (act !== null && act !== undefined) ? act : pred;
            
            if (val !== null && val !== undefined) {
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = i;
                }
                if (val < minVal) {
                    minVal = val;
                    minIdx = i;
                }
            }
        }
        
        if (maxIdx === -1 || minIdx === -1) return;
        
        // Draw highlights
        ctx.save();
        ctx.font = 'bold 10px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        
        const drawIndicator = (idx, val, label, color) => {
            const dsIndex = idx > 24 ? chart.data.datasets.indexOf(dsPredicted) : chart.data.datasets.indexOf(dsActual);
            if (dsIndex === -1) return;
            
            const meta = chart.getDatasetMeta(dsIndex);
            if (!meta || !meta.data || !meta.data[idx]) return;
            
            const x = meta.data[idx].x;
            const y = yAxis.getPixelForValue(val);
            
            // Draw glowing background dot
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, 2 * Math.PI);
            ctx.fillStyle = color + '22'; // 13% opacity
            ctx.fill();
            
            // Draw outer ring
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            
            // Draw center dot
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            
            // Draw a pill text badge above the point
            const timeLabel = chart.data.labels[idx];
            const hourOnly = timeLabel.split(' ').pop();
            const text = `${label}: ${val.toFixed(1)}°C @ ${hourOnly}`;
            ctx.font = 'bold 9px Outfit, sans-serif';
            const textWidth = ctx.measureText(text).width;
            const padX = 4;
            const padY = 2;
            const badgeW = textWidth + padX * 2;
            const badgeH = 12;
            const badgeX = x - badgeW / 2;
            const badgeY = y - badgeH - 5;
            
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 3);
            } else {
                ctx.rect(badgeX, badgeY, badgeW, badgeH);
            }
            ctx.fillStyle = 'rgba(17, 24, 39, 0.85)';
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = '#ffffff';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x, badgeY + badgeH / 2 + 0.5);
        };
        
        // Highlight MAX and MIN
        drawIndicator(maxIdx, maxVal, 'MAX', '#ef4444');
        drawIndicator(minIdx, minVal, 'MIN', '#3b82f6');
        
        ctx.restore();
    }
};

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
    } else if (currentPeriod === 'forecast' || currentPeriod === 'forecast_deviation') {
        const refNow = historyData.length > 0 ? historyData[historyData.length - 1].timestamp : Math.floor(Date.now() / 1000);
        const lastIndoor = historyData.length > 0 ? historyData[historyData.length - 1].temperature : 20.0;
        
        // Sync and render the forecast schedule table with the chart's current data and time anchor
        renderForecastSchedule(lastIndoor, refNow, historyData);
        
        // Round refNow to the nearest hour
        const nowHourTS = Math.round(refNow / 3600) * 3600;
        
        const numPastHours = 24;
        const numFutureHours = 24;
        const totalHours = numPastHours + numFutureHours + 1; // 49 hours total (24 past, now, 24 future)
        
        labels = new Array(totalHours);
        const actualIndoor = new Array(totalHours).fill(null);
        const predictedIndoor = new Array(totalHours).fill(null);
        const outdoorDataPoints = new Array(totalHours).fill(null);
        const outdoorCloudPoints = new Array(totalHours).fill(0);
        
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
                    if (outdoorForecast.hourly.cloud_cover) {
                        outdoorCloudPoints[idx] = outdoorForecast.hourly.cloud_cover[fIdx];
                    }
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
        
        // Pre-calculate effective outdoor temperatures adjusting for south-facing solar load
        const effectiveOutdoorData = new Array(totalHours).fill(null);
        for (let idx = 0; idx < totalHours; idx++) {
            const outTemp = outdoorDataPoints[idx];
            if (outTemp !== null) {
                const ts = nowHourTS + (idx - numPastHours) * 3600;
                const date = new Date(ts * 1000);
                const cloudCover = outdoorCloudPoints[idx];
                const solarBias = getSolarParameters(date, cloudCover);
                effectiveOutdoorData[idx] = parseFloat((outTemp + solarBias).toFixed(2));
            }
        }
        
        // 3. Connect predicted line to the last actual reading at index 24 (Now)
        predictedIndoor[numPastHours] = actualIndoor[numPastHours];
        
        // 4. Recursive thermal prediction for future hours using effective (solar-gain) outdoor temperatures
        const alpha = parseFloat(localStorage.getItem('optimized_insulation_rate') || '0.05');
        let currentSlope = getLatestTemperatureSlope(historyData);
        
        const slopeParamEl = document.getElementById('formula-param-slope');
        if (slopeParamEl) {
            slopeParamEl.innerHTML = `<strong>slope(t):</strong> Thermal momentum (currently <strong>${(currentSlope >= 0 ? '+' : '') + currentSlope.toFixed(2)}°C/h</strong>, decaying: &times;0.7/h)`;
        }
        for (let offset = 1; offset <= numFutureHours; offset++) {
            const idx = offset + numPastHours;
            const prevIdx = idx - 1;
            const prevIndoor = (offset === 1) ? actualIndoor[numPastHours] : predictedIndoor[prevIdx];
            const outTemp = effectiveOutdoorData[idx];
            
            if (prevIndoor !== null && outTemp !== null) {
                currentSlope *= 0.7; // Decay the momentum/slope factor hourly
                predictedIndoor[idx] = parseFloat((prevIndoor + alpha * (outTemp - prevIndoor) + 0.05 + currentSlope).toFixed(2));
            }
        }
        
        // 5. Apply dynamic bias correction to forecast points
        const biasCorrection = getHistoricalBiasCorrection();
        if (biasCorrection !== 0) {
            for (let offset = 1; offset <= numFutureHours; offset++) {
                const idx = offset + numPastHours;
                if (predictedIndoor[idx] !== null) {
                    predictedIndoor[idx] = parseFloat((predictedIndoor[idx] + biasCorrection).toFixed(2));
                }
            }
        }
        
        // Evaluate and save the forecast accuracy
        if (currentPeriod === 'forecast') {
            saveAndEvaluateForecast(historyData, predictedIndoor, numPastHours, refNow);
        }
           if (currentPeriod === 'forecast_deviation') {
            const simulatedPrediction = new Array(totalHours).fill(null);
            
            let firstValidIdx = -1;
            for (let idx = 0; idx <= numPastHours; idx++) {
                if (actualIndoor[idx] !== null) {
                    firstValidIdx = idx;
                    break;
                }
            }
            
            if (firstValidIdx !== -1) {
                simulatedPrediction[firstValidIdx] = actualIndoor[firstValidIdx];
                for (let idx = firstValidIdx + 1; idx <= numPastHours; idx++) {
                    const prevPred = simulatedPrediction[idx - 1];
                    const outTemp = effectiveOutdoorData[idx];
                    if (prevPred !== null && outTemp !== null) {
                        simulatedPrediction[idx] = prevPred + alpha * (outTemp - prevPred) + 0.05;
                    } else if (prevPred !== null) {
                        simulatedPrediction[idx] = prevPred + 0.05;
                    }
                }
            }
            
            const deviations = new Array(totalHours).fill(null);
            for (let idx = 0; idx < totalHours; idx++) {
                const inTemp = idx <= numPastHours ? simulatedPrediction[idx] : predictedIndoor[idx];
                const outTemp = effectiveOutdoorData[idx];
                if (inTemp !== null && outTemp !== null) {
                    deviations[idx] = parseFloat((outTemp - inTemp).toFixed(2));
                }
            }
            dataset1 = deviations;
            dataset2 = null;
            label1 = "Forecast Deviation (Outdoor Forecast/Measured - Closed-Window Prediction)";
        } else {
            dataset1 = actualIndoor;
            dataset2 = predictedIndoor;
            dataset3 = effectiveOutdoorData;
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
                const outTemp = effectiveOutdoorData[idx];
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
    if (currentPeriod === 'anomaly' || currentPeriod === 'forecast_deviation') {
        const numPastHours = 24;
        const borderColors = dataset1.map((v, idx) => {
            if (v === null) return 'transparent';
            if (currentPeriod === 'forecast_deviation' && idx > numPastHours) {
                return v >= 0 ? 'rgba(239, 68, 68, 0.45)' : 'rgba(59, 130, 246, 0.45)';
            }
            return v >= 0 ? '#ef4444' : '#3b82f6';
        });
        const backgroundColors = dataset1.map((v, idx) => {
            if (v === null) return 'rgba(0,0,0,0)';
            if (currentPeriod === 'forecast_deviation' && idx > numPastHours) {
                return v >= 0 ? 'rgba(239, 68, 68, 0.12)' : 'rgba(59, 130, 246, 0.12)';
            }
            return v >= 0 ? 'rgba(239, 68, 68, 0.35)' : 'rgba(59, 130, 246, 0.35)';
        });
        
        chartDatasets.push({
            label: label1,
            data: dataset1,
            borderColor: borderColors,
            backgroundColor: backgroundColors,
            borderWidth: 1.5,
            borderRadius: 4,
            yAxisID: 'y'
        });
    } else if (currentPeriod === 'forecast') {
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
            order: 1,
            pointStyle: 'line'
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
            order: 1,
            pointStyle: 'line'
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
            order: 1,
            pointStyle: 'line'
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
            pointBorderColor: color1,
            pointBorderWidth: 1.5,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: color1,
            pointHoverBorderColor: color1,
            pointHoverBorderWidth: 2,
            spanGaps: true,
            pointStyle: 'line'
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
            pointBorderColor: color2,
            pointBorderWidth: 1.5,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: color2,
            pointHoverBorderColor: color2,
            pointHoverBorderWidth: 2,
            spanGaps: true,
            pointStyle: 'line'
        });
    }

    if (chartInstance) {
        // Update the existing chart smoothly in-place without blinking
        chartInstance.data.labels = labels;
        
        if (currentPeriod === 'anomaly' || currentPeriod === 'forecast_deviation') {
            const numPastHours = 24;
            const borderColors = dataset1.map((v, idx) => {
                if (v === null) return 'transparent';
                if (currentPeriod === 'forecast_deviation' && idx > numPastHours) {
                    return v >= 0 ? 'rgba(239, 68, 68, 0.45)' : 'rgba(59, 130, 246, 0.45)';
                }
                return v >= 0 ? '#ef4444' : '#3b82f6';
            });
            const backgroundColors = dataset1.map((v, idx) => {
                if (v === null) return 'rgba(0,0,0,0)';
                if (currentPeriod === 'forecast_deviation' && idx > numPastHours) {
                    return v >= 0 ? 'rgba(239, 68, 68, 0.12)' : 'rgba(59, 130, 246, 0.12)';
                }
                return v >= 0 ? 'rgba(239, 68, 68, 0.35)' : 'rgba(59, 130, 246, 0.35)';
            });
            
            chartInstance.data.datasets[0].data = dataset1;
            chartInstance.data.datasets[0].label = label1;
            chartInstance.data.datasets[0].borderColor = borderColors;
            chartInstance.data.datasets[0].backgroundColor = backgroundColors;
        } else if (currentPeriod === 'forecast') {
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
            type: (currentPeriod === 'anomaly' || currentPeriod === 'forecast_deviation') ? 'bar' : 'line',
            data: {
                labels: labels,
                datasets: chartDatasets
            },
            plugins: [verticalLinePlugin, forecastExtremaPlugin],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 12 },
                            usePointStyle: true,
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
                                const label = context.dataset.label || '';
                                if (label === 'Action Suggestion') {
                                    return null;
                                }
                                const val = context.parsed.y;
                                 if (currentPeriod === 'anomaly' || currentPeriod === 'forecast_deviation') {
                                     return `Deviation: ${val >= 0 ? '+' : ''}${val.toFixed(2)}°C`;
                                 }
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
                                 if (currentPeriod === 'anomaly' || currentPeriod === 'forecast_deviation') {
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

// Custom plugin to draw horizontal bands representing comfort/dry/humid zones
const humidityBandsPlugin = {
    id: 'humidityBands',
    beforeDraw: (chart) => {
        if (currentPeriod === 'anomaly') return;
        const ctx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        if (!yAxis) return;
        
        ctx.save();
        
        const y60 = yAxis.getPixelForValue(60);
        const y30 = yAxis.getPixelForValue(30);
        const yTop = yAxis.top;
        const yBottom = yAxis.bottom;
        const xLeft = xAxis.left;
        const xRight = xAxis.right;
        
        // Humid zone (> 60% RH) - soft blue/cyan
        if (y60 > yTop) {
            ctx.fillStyle = 'rgba(59, 130, 246, 0.04)'; 
            ctx.fillRect(xLeft, yTop, xRight - xLeft, y60 - yTop);
            
            ctx.fillStyle = 'rgba(59, 130, 246, 0.35)';
            ctx.font = '500 10px Outfit, sans-serif';
            ctx.fillText('HUMID (>60%)', xLeft + 10, yTop + 14);
        }
        
        // Ideal zone (30% - 60% RH) - soft green
        const idealTop = Math.max(y60, yTop);
        const idealBottom = Math.min(y30, yBottom);
        if (idealBottom > idealTop) {
            ctx.fillStyle = 'rgba(16, 185, 129, 0.03)'; 
            ctx.fillRect(xLeft, idealTop, xRight - xLeft, idealBottom - idealTop);
            
            ctx.fillStyle = 'rgba(16, 185, 129, 0.35)';
            ctx.font = '500 10px Outfit, sans-serif';
            ctx.fillText('IDEAL (30%-60%)', xLeft + 10, idealTop + 14);
        }
        
        // Dry zone (< 30% RH) - soft orange
        if (y30 < yBottom) {
            ctx.fillStyle = 'rgba(245, 158, 11, 0.04)'; 
            ctx.fillRect(xLeft, y30, xRight - xLeft, yBottom - y30);
            
            ctx.fillStyle = 'rgba(245, 158, 11, 0.35)';
            ctx.font = '500 10px Outfit, sans-serif';
            ctx.fillText('DRY (<30%)', xLeft + 10, y30 + 14);
        }
        
        ctx.restore();
    }
};

// Draw or update the Chart.js line graph for Relative Humidity
function drawHumidityChart(historyData) {
    if (humidityChartInstance && activeHumidityChartPeriod !== currentPeriod) {
        humidityChartInstance.destroy();
        humidityChartInstance = null;
    }
    
    const canvas = document.getElementById('humidityChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    let labels, dataset1, dataset2;
    let label1, label2;
    let color1 = '#3b82f6'; // Sleek blue for humidity
    let color2 = '#60a5fa'; // Lighter blue for compare/yesterday
    
    if (currentPeriod === 'compare' || currentPeriod === 'anomaly') {
        const now = historyData.length > 0 ? historyData[historyData.length - 1].timestamp : Math.floor(Date.now() / 1000);
        const numBuckets = 288; // 24 hours / 5 min
        const bucketSize = 300; // 5 min in seconds

        const todayHums = new Array(numBuckets).fill(null);
        const yesterdayHums = new Array(numBuckets).fill(null);
        labels = new Array(numBuckets);

        for (let i = 0; i < numBuckets; i++) {
            const ts = now - i * bucketSize;
            labels[numBuckets - 1 - i] = formatTimestamp(ts, '24h');
        }

        historyData.forEach(d => {
            const age = now - d.timestamp;
            if (d.humidity !== null && d.humidity !== undefined) {
                if (age >= 0 && age < 86400) {
                    const bucketIdx = Math.floor(age / bucketSize);
                    if (bucketIdx >= 0 && bucketIdx < numBuckets) {
                        todayHums[numBuckets - 1 - bucketIdx] = d.humidity;
                    }
                } else if (age >= 86400 && age < 2 * 86400) {
                    const bucketIdx = Math.floor((age - 86400) / bucketSize);
                    if (bucketIdx >= 0 && bucketIdx < numBuckets) {
                        yesterdayHums[numBuckets - 1 - bucketIdx] = d.humidity;
                    }
                }
            }
        });
        
        if (currentPeriod === 'anomaly') {
            const anomalies = new Array(numBuckets).fill(null);
            for (let i = 0; i < numBuckets; i++) {
                if (todayHums[i] !== null && yesterdayHums[i] !== null) {
                    anomalies[i] = parseFloat((todayHums[i] - yesterdayHums[i]).toFixed(2));
                }
            }
            dataset1 = anomalies;
            dataset2 = null;
            label1 = "Today's Humidity Deviation from Yesterday";
        } else {
            dataset1 = todayHums;
            dataset2 = yesterdayHums;
            label1 = "Today's Humidity";
            label2 = "Yesterday's Humidity";
        }
    } else if (currentPeriod === 'forecast' || currentPeriod === 'forecast_deviation') {
        const refNow = historyData.length > 0 ? historyData[historyData.length - 1].timestamp : Math.floor(Date.now() / 1000);
        const nowHourTS = Math.round(refNow / 3600) * 3600;
        
        const numPastHours = 24;
        const numFutureHours = 24;
        const totalHours = numPastHours + numFutureHours + 1; // 49 hours total
        
        labels = new Array(totalHours);
        const actualIndoor = new Array(totalHours).fill(null);
        
        for (let idx = 0; idx < totalHours; idx++) {
            const hourOffset = idx - numPastHours;
            const hourTS = nowHourTS + hourOffset * 3600;
            labels[idx] = formatTimestamp(hourTS, 'forecast');
        }
        
        historyData.forEach(d => {
            if (d.humidity !== null && d.humidity !== undefined) {
                const diffSecs = d.timestamp - nowHourTS;
                const hourOffset = Math.round(diffSecs / 3600);
                const idx = hourOffset + numPastHours;
                if (idx >= 0 && idx < totalHours) {
                    actualIndoor[idx] = d.humidity;
                }
            }
        });
        
        dataset1 = actualIndoor;
        dataset2 = null;
        label1 = 'Indoor Humidity (Actual)';
    } else {
        labels = historyData.map(d => formatTimestamp(d.timestamp, currentPeriod));
        dataset1 = historyData.map(d => d.humidity);
        dataset2 = null;
        label1 = 'Relative Humidity';
    }
    
    const humGradient = ctx.createLinearGradient(0, 0, 0, 300);
    humGradient.addColorStop(0, hexToRgbA(color1, 0.25));
    humGradient.addColorStop(0.5, hexToRgbA(color1, 0.08));
    humGradient.addColorStop(1, hexToRgbA(color1, 0.0));

    const compareGradient = ctx.createLinearGradient(0, 0, 0, 300);
    compareGradient.addColorStop(0, hexToRgbA(color2, 0.25));
    compareGradient.addColorStop(0.5, hexToRgbA(color2, 0.08));
    compareGradient.addColorStop(1, hexToRgbA(color2, 0.0));

    const chartDatasets = [];
    if (currentPeriod === 'anomaly') {
        const borderColors = dataset1.map(v => {
            if (v === null) return 'transparent';
            return v >= 0 ? '#ef4444' : '#3b82f6';
        });
        const backgroundColors = dataset1.map(v => {
            if (v === null) return 'rgba(0,0,0,0)';
            return v >= 0 ? 'rgba(239, 68, 68, 0.35)' : 'rgba(59, 130, 246, 0.35)';
        });
        
        chartDatasets.push({
            label: label1,
            data: dataset1,
            borderColor: borderColors,
            backgroundColor: backgroundColors,
            borderWidth: 1.5,
            borderRadius: 4
        });
    } else {
        chartDatasets.push({
            label: label1,
            data: dataset1,
            borderColor: color1,
            backgroundColor: humGradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            pointRadius: (context) => {
                const count = context.chart.data.datasets[0].data.length;
                return count < 40 ? 3.5 : 0;
            },
            pointBackgroundColor: color1,
            pointBorderColor: color1,
            pointBorderWidth: 1.5,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: color1,
            pointHoverBorderColor: color1,
            pointHoverBorderWidth: 2,
            spanGaps: true,
            pointStyle: 'line'
        });
        
        if (dataset2) {
            chartDatasets.push({
                label: label2,
                data: dataset2,
                borderColor: color2,
                backgroundColor: compareGradient,
                borderWidth: 2.5,
                fill: true,
                tension: 0.35,
                pointRadius: (context) => {
                    const count = context.chart.data.datasets[1].data.length;
                    return count < 40 ? 3.5 : 0;
                },
                pointBackgroundColor: color2,
                pointBorderColor: color2,
                pointBorderWidth: 1.5,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: color2,
                pointHoverBorderColor: color2,
                pointHoverBorderWidth: 2,
                spanGaps: true,
                pointStyle: 'line'
            });
        }
    }
    
    if (humidityChartInstance) {
        humidityChartInstance.data.labels = labels;
        humidityChartInstance.data.datasets = chartDatasets;
        humidityChartInstance.update('none');
    } else {
        activeHumidityChartPeriod = currentPeriod;
        humidityChartInstance = new Chart(ctx, {
            type: (currentPeriod === 'anomaly') ? 'bar' : 'line',
            data: {
                labels: labels,
                datasets: chartDatasets
            },
            plugins: [humidityBandsPlugin],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 12 },
                            usePointStyle: true
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
                                const label = context.dataset.label || '';
                                const val = context.parsed.y;
                                if (currentPeriod === 'anomaly') {
                                    return `Deviation: ${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
                                }
                                return `${label}: ${val !== null && val !== undefined ? val.toFixed(1) : '--.-'}%`;
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
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 11 },
                            callback: function(value) {
                                if (currentPeriod === 'anomaly') {
                                    return (value >= 0 ? '+' : '') + value.toFixed(0) + '%';
                                }
                                return value.toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Helper to calculate the latest temperature slope (°C/hour)
function getLatestTemperatureSlope(historyData) {
    if (!historyData || historyData.length < 2) return 0;
    
    // Sort historyData by timestamp ascending just in case
    const sortedData = [...historyData].sort((a, b) => a.timestamp - b.timestamp);
    const latest = sortedData[sortedData.length - 1];
    const latestTime = latest.timestamp;
    
    // Filter data points from the last 1 hour (3600 seconds)
    const oneHourAgo = latestTime - 3600;
    let points = sortedData.filter(d => d.timestamp >= oneHourAgo);
    
    if (points.length < 2) {
        // Fallback to last 2 hours
        const twoHoursAgo = latestTime - 7200;
        points = sortedData.filter(d => d.timestamp >= twoHoursAgo);
        if (points.length < 2) return 0;
    }
    
    const n = points.length;
    const t0 = points[0].timestamp;
    
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < n; i++) {
        // Convert timestamp to hours relative to t0
        const x = (points[i].timestamp - t0) / 3600; 
        const y = points[i].temperature;
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }
    
    const denom = (n * sumX2 - sumX * sumX);
    let slope = 0; // °C per hour
    if (denom !== 0) {
        slope = (n * sumXY - sumX * sumY) / denom;
    } else {
        const dt = (points[n - 1].timestamp - points[0].timestamp) / 3600;
        if (dt > 0) {
            slope = (points[n - 1].temperature - points[0].temperature) / dt;
        }
    }
    return slope;
}

// Estimate the temperature gradient and update the dashboard trend indicator
function updateTemperatureGradient(historyData) {
    const trendContainer = document.getElementById('temp-trend-container');
    const trendArrow = document.getElementById('temp-trend-arrow');
    const trendValue = document.getElementById('temp-trend-value');
    if (!trendContainer || !trendArrow || !trendValue || !historyData || historyData.length < 2) {
        if (trendContainer) trendContainer.style.display = 'none';
        return;
    }
    
    const slope = getLatestTemperatureSlope(historyData);
    
    // Choose trend arrow based on threshold (steady if change is within +/- 0.1°C/h)
    const threshold = 0.1; 
    let arrow = '→';
    let arrowClass = 'steady';
    
    if (slope > threshold) {
        arrow = '↑';
        arrowClass = 'rising';
    } else if (slope < -threshold) {
        arrow = '↓';
        arrowClass = 'falling';
    }
    
    trendArrow.textContent = arrow;
    trendArrow.className = `trend-arrow ${arrowClass}`;
    
    const sign = slope >= 0 ? '+' : '';
    trendValue.textContent = `${sign}${slope.toFixed(2)}°C/h`;
    
    trendContainer.style.display = 'flex';
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
        
        if (period === 'forecast' || period === 'forecast_deviation') {
            fetchPeriod = '24h';
            // Fetch outdoor forecast from Open-Meteo for Rue Sarrette coordinates
            try {
                const url = `https://api.open-meteo.com/v1/forecast?latitude=${LATITUDE}&longitude=${LONGITUDE}&hourly=temperature_2m,cloud_cover&timezone=auto`;
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
        
        if (period === 'forecast' || period === 'forecast_deviation') {
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
        
        // Update the live slope parameter inside the equation legend on load
        const currentSlope = getLatestTemperatureSlope(historyData);
        const slopeParamEl = document.getElementById('formula-param-slope');
        if (slopeParamEl) {
            slopeParamEl.innerHTML = `<strong>slope(t):</strong> Thermal momentum (currently <strong>${(currentSlope >= 0 ? '+' : '') + currentSlope.toFixed(2)}°C/h</strong>, decaying: &times;0.7/h)`;
        }
        
        // Run insulation optimization and update UI
        let alpha = 0.05;
        if (historyData && historyData.length > 0) {
            alpha = optimizeInsulationRate(historyData);
        }
        const biasCorrection = getHistoricalBiasCorrection();
        updateFormulaUI(alpha, currentSlope, biasCorrection);
        
        drawChart(historyData);
        drawHumidityChart(historyData);
        updateTemperatureGradient(historyData);
        
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

// Render Forecast Schedule (Next 8 Hours) synced with the given reference time and indoor temp
function renderForecastSchedule(lastIndoor, referenceTimestamp, historyData) {
    const scheduleBody = document.getElementById('forecast-schedule-body');
    if (scheduleBody && outdoorForecast && outdoorForecast.hourly) {
        scheduleBody.innerHTML = '';
        const nowHourTS = Math.round(referenceTimestamp / 3600) * 3600;
        const biasCorrection = getHistoricalBiasCorrection();
        
        const getIsoHourString = (d) => {
            const yyyy = d.getFullYear();
            const mm = String(d.getMonth() + 1).padStart(2, '0');
            const dd = String(d.getDate()).padStart(2, '0');
            const hh = String(d.getHours()).padStart(2, '0');
            return `${yyyy}-${mm}-${dd}T${hh}:00`;
        };
        
        // Loop for the next 8 hours (starting at the current hour)
        let currentPred = lastIndoor;
        let currentSlope = getLatestTemperatureSlope(historyData);
        for (let h = 0; h < 8; h++) {
            const ts = nowHourTS + h * 3600;
            const dateObj = new Date(ts * 1000);
            const pad = (n) => String(n).padStart(2, '0');
            const timeStr = `${pad(dateObj.getHours())}:${pad(dateObj.getMinutes())}`;
            
            let outTemp = null;
            let cloudCover = 0;
            const iso = getIsoHourString(dateObj);
            const fIdx = outdoorForecast.hourly.time.indexOf(iso);
            if (fIdx !== -1) {
                outTemp = outdoorForecast.hourly.temperature_2m[fIdx];
                if (outdoorForecast.hourly.cloud_cover) {
                    cloudCover = outdoorForecast.hourly.cloud_cover[fIdx];
                }
            }
            
            if (outTemp !== null) {
                const solarBias = getSolarParameters(dateObj, cloudCover);
                const effectiveOut = outTemp + solarBias;
                
                if (h > 0) {
                    currentSlope *= 0.7; // Decay slope over time
                    // Predict step only for future hours using effective outdoor temperature (with solar bias) and slope
                    currentPred = currentPred + 0.05 * (effectiveOut - currentPred) + 0.05 + currentSlope;
                }
                
                // Apply bias correction to future predictions
                const correctedPred = h > 0 ? (currentPred + biasCorrection) : currentPred;
                
                // Calculate delta using effective outdoor temp to align with the Forecast Deviation plot
                const delta = effectiveOut - correctedPred;
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
                
                if (correctedPred < COMFORT_MIN) {
                    // Indoor is too cold: open only if outdoor is warmer than indoor to help heat the room
                    isOpen = effectiveOut > correctedPred;
                } else if (correctedPred > COMFORT_MAX) {
                    // Indoor is too hot: open only if outdoor is cooler than indoor to help cool the room
                    isOpen = effectiveOut < correctedPred;
                } else {
                    // Indoor is in comfort range: open only if outdoor air is also comfortable to maintain state
                    isOpen = effectiveOut >= COMFORT_MIN && effectiveOut <= COMFORT_MAX;
                }
                
                const verdict = isOpen 
                    ? '<span style="color:var(--success);font-weight:bold;">🔓 OPEN</span>' 
                    : '<span style="color:var(--danger);font-weight:bold;">🔒 CLOSE</span>';
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${timeStr}</strong></td>
                    <td>${deltaHtml}</td>
                    <td>${verdict}</td>
                `;
                scheduleBody.appendChild(tr);
            }
        }
    }
}

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

    // 1.5. Render Forecast Schedule (Next 8 Hours)
    const lastRecord = history7d[history7d.length - 1];
    if (lastRecord) {
        renderForecastSchedule(lastRecord.temperature, lastRecord.timestamp, history7d);
    }
    
    // 2. Determine Best Airing/Forecast Time (coolest hour)
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
    
    const forecastEl = document.getElementById('insight-forecast');
    if (forecastEl) {
        const startHour = parseInt(coolestHour);
        const endHour = (startHour + 2) % 24;
        const pad = (h) => String(h).padStart(2, '0');
        forecastEl.innerHTML = `Open windows between <strong>${pad(startHour)}:00 and ${pad(endHour)}:00</strong> when indoor temperature drops to its daily low (avg <strong>${minHourlyAvg.toFixed(1)}°C</strong>).`;
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

// Run parameter optimization on 24h history to find the best-fitting insulation rate (alpha)
function optimizeInsulationRate(historyData) {
    const validPoints = historyData.filter(d => d.temperature !== null && d.outdoorTemperature !== null);
    if (validPoints.length < 12) return 0.05; // Fallback to default if insufficient data
    
    let bestAlpha = 0.05;
    let minMSE = Infinity;
    
    // Sort points chronologically
    const sortedPoints = [...validPoints].sort((a, b) => a.timestamp - b.timestamp);
    
    // Test alpha values from 0.01 to 0.15 in steps of 0.005
    for (let alpha = 0.01; alpha <= 0.15; alpha += 0.005) {
        let sumSquaredError = 0;
        let count = 0;
        
        let simulatedT = sortedPoints[0].temperature;
        for (let i = 1; i < sortedPoints.length; i++) {
            const dt = (sortedPoints[i].timestamp - sortedPoints[i - 1].timestamp) / 3600; // time gap in hours
            const outTemp = sortedPoints[i].outdoorTemperature;
            
            // Recursive prediction step based on the convective model (scaled by dt)
            simulatedT = simulatedT + alpha * dt * (outTemp - simulatedT) + 0.05 * dt;
            
            const actualT = sortedPoints[i].temperature;
            sumSquaredError += Math.pow(simulatedT - actualT, 2);
            count++;
        }
        
        const mse = sumSquaredError / count;
        if (mse < minMSE) {
            minMSE = mse;
            bestAlpha = alpha;
        }
    }
    
    const optimizedVal = parseFloat(bestAlpha.toFixed(4));
    localStorage.setItem('optimized_insulation_rate', optimizedVal.toString());
    return optimizedVal;
}

// Get a natural language interpretation of the convective insulation rate coefficient
function getInsulationInterpretation(alpha) {
    if (alpha <= 0.03) {
        return "Excellent insulation";
    } else if (alpha <= 0.06) {
        return "Good standard insulation";
    } else if (alpha <= 0.09) {
        return "Moderate/Drafty insulation";
    } else {
        return "Poor insulation / open windows";
    }
}

// Update the parameter equation text inside the details summary block
function updateFormulaUI(alpha, slope, bias) {
    const mathBox = document.getElementById('formula-math-box');
    if (mathBox) {
        mathBox.innerHTML = `T<sub>in</sub>(t) = T<sub>in</sub>(t-1) + <strong>${alpha.toFixed(3)}</strong> &times; [T<sub>out</sub>(t) - T<sub>in</sub>(t-1)] + 0.05 + slope(t) + bias`;
    }
    
    const interpretation = getInsulationInterpretation(alpha);
    const alphaParamEl = document.getElementById('formula-param-alpha');
    if (alphaParamEl) {
        alphaParamEl.innerHTML = `<strong>${alpha.toFixed(3)}:</strong> Convective transfer insulation rate (optimized: <strong>${interpretation}</strong>)`;
    }
}

// Helper to save current forecast and evaluate the accuracy of previous forecasts
async function saveAndEvaluateForecast(historyData, predictedIndoor, numPastHours, refNow) {
    const lastForecastStr = localStorage.getItem('last_forecast');
    if (lastForecastStr) {
        try {
            const lastForecast = JSON.parse(lastForecastStr);
            const timeDiff = refNow - lastForecast.timestamp;
            
            // We evaluate if at least 24 hours have elapsed
            if (timeDiff >= 24 * 3600) {
                const evalStart = lastForecast.timestamp;
                const evalEnd = lastForecast.timestamp + 24 * 3600;
                
                // Find actual temperatures in this 24-hour window
                const actualsInRange = historyData
                    .filter(d => d.timestamp >= evalStart && d.timestamp <= evalEnd)
                    .map(d => d.temperature);
                
                // Only evaluate if we have sufficient readings in that 24h period (at least 12 readings)
                if (actualsInRange.length >= 12) {
                    const actualMax = Math.max(...actualsInRange);
                    const actualMin = Math.min(...actualsInRange);
                    
                    const errMax = actualMax - lastForecast.predictedMax;
                    const errMin = actualMin - lastForecast.predictedMin;
                    
                    // POST the evaluation result to the server
                    try {
                        const response = await fetch('/api/log-forecast-error', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                timestamp: refNow,
                                errMax: parseFloat(errMax.toFixed(2)),
                                errMin: parseFloat(errMin.toFixed(2))
                            })
                        });
                        
                        if (response.ok) {
                            localStorage.removeItem('last_forecast'); // Remove so we don't re-evaluate
                            // Re-fetch forecast errors list to update global state and UI
                            await fetchForecastErrors();
                        }
                    } catch (apiErr) {
                        console.error("Failed to POST forecast error to server:", apiErr);
                    }
                }
            }
        } catch (e) {
            console.error("Error evaluating past forecast:", e);
        }
    }
    
    // Save current forecast if we don't have an active one (or if we just evaluated the last one)
    if (!localStorage.getItem('last_forecast')) {
        const futurePredictions = predictedIndoor.slice(numPastHours).filter(v => v !== null && v !== undefined);
        if (futurePredictions.length > 0) {
            const predictedMax = Math.max(...futurePredictions);
            const predictedMin = Math.min(...futurePredictions);
            
            const newForecast = {
                timestamp: refNow,
                predictedMax: parseFloat(predictedMax.toFixed(2)),
                predictedMin: parseFloat(predictedMin.toFixed(2))
            };
            localStorage.setItem('last_forecast', JSON.stringify(newForecast));
        }
    }
    
    renderAccuracyUI();
}

// Fetch forecast errors from the server
async function fetchForecastErrors() {
    try {
        const response = await fetch('/api/forecast-errors');
        if (response.ok) {
            forecastErrorsList = await response.json();
            renderAccuracyUI();
        }
    } catch (err) {
        console.error("Error fetching forecast errors list:", err);
    }
}

// Helper to get the average historical forecast error (mean error) to correct predictions dynamically
function getHistoricalBiasCorrection() {
    if (forecastErrorsList.length < 3) return 0; // Require at least 3 evaluations to avoid premature adjustments
    
    let sumMaxErr = 0;
    let sumMinErr = 0;
    forecastErrorsList.forEach(e => {
        sumMaxErr += e.errMax;
        sumMinErr += e.errMin;
    });
    
    const avgMaxErr = sumMaxErr / forecastErrorsList.length;
    const avgMinErr = sumMinErr / forecastErrorsList.length;
    
    // Average the peak and low errors to find the general shift
    const correction = (avgMaxErr + avgMinErr) / 2;
    
    // Cap correction to a maximum of +/- 2.0°C to keep predictions stable and safe
    return Math.max(-2.0, Math.min(2.0, correction));
}

// Render the accuracy metrics in the UI
function renderAccuracyUI() {
    const accuracyEl = document.getElementById('insight-accuracy');
    if (!accuracyEl) return;
    
    if (forecastErrorsList.length === 0) {
        accuracyEl.innerHTML = `Model accuracy tracking started. Average error metrics will appear after 24 hours of data logging.`;
        return;
    }
    
    let sumAbsMaxErr = 0;
    let sumAbsMinErr = 0;
    forecastErrorsList.forEach(e => {
        sumAbsMaxErr += Math.abs(e.errMax);
        sumAbsMinErr += Math.abs(e.errMin);
    });
    
    const avgMaxErr = sumAbsMaxErr / forecastErrorsList.length;
    const avgMinErr = sumAbsMinErr / forecastErrorsList.length;
    
    const biasCorrection = getHistoricalBiasCorrection();
    let biasInfo = '';
    if (biasCorrection !== 0) {
        const direction = biasCorrection > 0 ? 'adding' : 'subtracting';
        biasInfo = `<br><span style="font-size:0.8rem;color:var(--text-muted);">Dynamic bias correction active: ${direction} <strong>${Math.abs(biasCorrection).toFixed(2)}°C</strong> to predictions.</span>`;
    }
    
    // Update the live bias parameter in the equation legend
    const biasParamEl = document.getElementById('formula-param-bias');
    if (biasParamEl) {
        biasParamEl.innerHTML = `<strong>bias:</strong> Dynamic offset correction (currently <strong>${(biasCorrection >= 0 ? '+' : '') + biasCorrection.toFixed(2)}°C</strong>, derived from error logs)`;
    }
    
    accuracyEl.innerHTML = `Over the last <strong>${forecastErrorsList.length}</strong> runs, the 24h forecast model is accurate within <strong style="color:#ef4444;">±${avgMaxErr.toFixed(2)}°C</strong> for peaks and <strong style="color:#3b82f6;">±${avgMinErr.toFixed(2)}°C</strong> for daily lows.${biasInfo}`;
}

// Initialization
async function init() {
    await fetchForecastErrors(); // Initial fetch of forecast errors from database
    
    // Initial fetch of status and temperature
    await fetchCurrentTemp();
    
    // Fetch outdoor forecast immediately on startup for Rue Sarrette to ensure it's available for insights
    try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${LATITUDE}&longitude=${LONGITUDE}&hourly=temperature_2m,cloud_cover&timezone=auto`;
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
