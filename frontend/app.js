// Constants
const API_BASE_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
    ? 'http://127.0.0.1:8000' 
    : window.location.origin;

// Global variables
let selectedFile = null;
let parsedBatchResults = null;

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCharCounter();
    initSingleAnalysis();
    initBatchAnalysis();
    initMetricsTab();
    loadHistory();
});

// 1. Navigation Tab Switching Logic
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) targetPane.classList.add('active');

            if (targetTab === 'model-metrics') {
                fetchMetrics();
            }
        });
    });
}

// 2. Character Counter for Review Textarea
function initCharCounter() {
    const textarea = document.getElementById('review-input');
    const charCurrent = document.getElementById('char-current');

    if (textarea && charCurrent) {
        textarea.addEventListener('input', () => {
            charCurrent.textContent = textarea.value.length.toLocaleString();
        });
    }
}

// 3. Single Review Analysis Inference
function initSingleAnalysis() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const reviewInput = document.getElementById('review-input');
    const resultsCard = document.getElementById('results-card');
    const analysisLoader = document.getElementById('analysis-loader');
    const resultsContent = document.getElementById('results-content');

    if (!analyzeBtn) return;

    analyzeBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        if (text.length < 3) {
            showToast("Review text must be at least 3 characters long.");
            return;
        }

        // Setup Loading State
        resultsCard.classList.remove('empty-state');
        analysisLoader.classList.remove('hidden');
        resultsContent.classList.add('hidden');

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Server error running predictions.");
            }

            const data = await response.json();
            renderResults(data);
            saveToHistory(data);
        } catch (error) {
            console.error("Error analyzing sentiment:", error);
            showToast(error.message || "Connection refused by backend service.");
            resultsCard.classList.add('empty-state');
            analysisLoader.classList.add('hidden');
        } finally {
            analysisLoader.classList.add('hidden');
        }
    });

    clearBtn.addEventListener('click', () => {
        reviewInput.value = '';
        document.getElementById('char-current').textContent = '0';
        resultsCard.classList.add('empty-state');
        resultsContent.classList.add('hidden');
        analysisLoader.classList.add('hidden');
    });
}

function renderResults(data) {
    const resultsContent = document.getElementById('results-content');
    const overallBadge = document.getElementById('overall-badge');
    
    // LR Elements
    const lrSentimentVal = document.getElementById('lr-sentiment-val');
    const lrConfidenceVal = document.getElementById('lr-confidence-val');
    const lrProgressFill = document.getElementById('lr-progress-fill');
    const lrIndicator = document.getElementById('lr-sentiment-indicator');
    const lrIcon = document.getElementById('lr-sentiment-icon');

    // LSTM Elements
    const lstmSentimentVal = document.getElementById('lstm-sentiment-val');
    const lstmConfidenceVal = document.getElementById('lstm-confidence-val');
    const lstmProgressFill = document.getElementById('lstm-progress-fill');
    const lstmIndicator = document.getElementById('lstm-sentiment-indicator');
    const lstmIcon = document.getElementById('lstm-sentiment-icon');

    // Overall Consensus (Prefer Deep Learning LSTM, fallback to LR if they match)
    const consensusSentiment = data.lstm.sentiment;
    overallBadge.textContent = consensusSentiment;
    overallBadge.className = `badge ${consensusSentiment.toLowerCase()}`;

    // Update LR prediction display
    updateModelResultDisplay(data.lr, lrSentimentVal, lrConfidenceVal, lrProgressFill, lrIndicator, lrIcon);

    // Update LSTM prediction display
    updateModelResultDisplay(data.lstm, lstmSentimentVal, lstmConfidenceVal, lstmProgressFill, lstmIndicator, lstmIcon);

    resultsContent.classList.remove('hidden');
}

function updateModelResultDisplay(prediction, sentimentLabel, confidenceLabel, progressBar, indicator, icon) {
    sentimentLabel.textContent = prediction.sentiment;
    confidenceLabel.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
    progressBar.style.width = `${(prediction.confidence * 100)}%`;

    // Apply class formatting
    indicator.className = `sentiment-indicator ${prediction.sentiment.toLowerCase()}`;
    icon.className = `fa-solid ${
        prediction.sentiment === 'Positive' ? 'fa-face-smile' : 
        prediction.sentiment === 'Negative' ? 'fa-face-frown' : 'fa-face-meh'
    }`;
}

// 4. Batch Analysis file uploading and processing
function initBatchAnalysis() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('csv-file-input');
    const fileDetails = document.getElementById('file-details');
    const fileNameText = document.getElementById('selected-file-name');
    const fileSizeText = document.getElementById('selected-file-size');
    const cancelFileBtn = document.getElementById('cancel-file-btn');
    const batchAnalyzeBtn = document.getElementById('batch-analyze-btn');
    
    const resultsWrapper = document.getElementById('batch-results-wrapper');
    const batchLoader = document.getElementById('batch-loader');
    const resultsContent = document.getElementById('batch-results-content');
    const downloadCsvBtn = document.getElementById('download-csv-btn');

    // Click trigger on upload zone
    uploadZone.addEventListener('click', () => fileInput.click());

    // Drag-over styling classes
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelected(files[0]);
        }
    });

    // Handle file changes via input browse
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelected(e.target.files[0]);
        }
    });

    function handleFileSelected(file) {
        if (!file.name.endsWith('.csv')) {
            showToast("Only CSV files are supported.");
            return;
        }
        selectedFile = file;
        fileNameText.textContent = file.name;
        fileSizeText.textContent = `${(file.size / 1024).toFixed(1)} KB`;
        
        uploadZone.classList.add('hidden');
        fileDetails.classList.remove('hidden');
    }

    cancelFileBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        uploadZone.classList.remove('hidden');
        fileDetails.classList.add('hidden');
        resultsWrapper.classList.add('hidden');
    });

    batchAnalyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        resultsWrapper.classList.remove('hidden');
        batchLoader.classList.remove('hidden');
        resultsContent.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE_URL}/predict-batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Server error processing batch file.");
            }

            const data = await response.json();
            parsedBatchResults = data.results;
            renderBatchResults(data.results);
        } catch (error) {
            console.error("Error running batch analysis:", error);
            showToast(error.message || "Failed to complete batch analysis.");
            resultsWrapper.classList.add('hidden');
        } finally {
            batchLoader.classList.add('hidden');
        }
    });

    downloadCsvBtn.addEventListener('click', () => {
        if (!parsedBatchResults || parsedBatchResults.length === 0) return;
        downloadCSVResults(parsedBatchResults);
    });
}

function renderBatchResults(results) {
    const tableBody = document.getElementById('batch-results-body');
    const totalCountText = document.getElementById('batch-total-count');
    const resultsContent = document.getElementById('batch-results-content');

    tableBody.innerHTML = '';
    totalCountText.textContent = results.length;

    results.forEach(res => {
        const tr = document.createElement('tr');
        
        // Truncate review column content for readable tables
        const displayReview = res.review.length > 100 ? res.review.substring(0, 97) + '...' : res.review;

        tr.innerHTML = `
            <td title="${res.review}">${escapeHtml(displayReview)}</td>
            <td class="sentiment-cell ${res.lr_sentiment.substring(0, 3).toLowerCase()}">${res.lr_sentiment}</td>
            <td>${(res.lr_confidence * 100).toFixed(0)}%</td>
            <td class="sentiment-cell ${res.lstm_sentiment.substring(0, 3).toLowerCase()}">${res.lstm_sentiment}</td>
            <td>${(res.lstm_confidence * 100).toFixed(0)}%</td>
        `;
        tableBody.appendChild(tr);
    });

    resultsContent.classList.remove('hidden');
}

function downloadCSVResults(results) {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Review,LR_Sentiment,LR_Confidence,LSTM_Sentiment,LSTM_Confidence\n";

    results.forEach(res => {
        // Escape quotes in reviews
        const cleanReview = res.review.replace(/"/g, '""');
        csvContent += `"${cleanReview}",${res.lr_sentiment},${res.lr_confidence},${res.lstm_sentiment},${res.lstm_confidence}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `sentifilm_analysis_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 5. Performance Metrics API Dashboard Loading
function initMetricsTab() {
    // Fetches metrics on clicking tab (handled by click listener on tabs)
}

async function fetchMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (!response.ok) throw new Error("Could not load metrics.");
        const data = await response.json();
        
        // Render LR metrics
        renderModelMetrics('lr', data.lr);
        // Render LSTM metrics
        renderModelMetrics('lstm', data.lstm);
    } catch (error) {
        console.error("Error loading metrics dashboard:", error);
        showToast("Model evaluation metrics are not available yet. Please complete training first.");
    }
}

function renderModelMetrics(prefix, metrics) {
    document.getElementById(`${prefix}-metric-acc`).textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById(`${prefix}-metric-prec`).textContent = `${(metrics.precision * 100).toFixed(1)}%`;
    document.getElementById(`${prefix}-metric-rec`).textContent = `${(metrics.recall * 100).toFixed(1)}%`;
    document.getElementById(`${prefix}-metric-f1`).textContent = `${(metrics.f1 * 100).toFixed(1)}%`;

    // Confusion Matrix mapping (assuming positive=1, negative=0)
    // cm is [[tn, fp], [fn, tp]]
    const cm = metrics.confusion_matrix;
    document.getElementById(`${prefix}-cm-tn`).textContent = cm[0][0].toLocaleString();
    document.getElementById(`${prefix}-cm-fp`).textContent = cm[0][1].toLocaleString();
    document.getElementById(`${prefix}-cm-fn`).textContent = cm[1][0].toLocaleString();
    document.getElementById(`${prefix}-cm-tp`).textContent = cm[1][1].toLocaleString();
}

// 6. History LocalStorage Management
function loadHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;

    const history = JSON.parse(localStorage.getItem('sentifilm_history') || '[]');
    if (history.length === 0) {
        historyList.innerHTML = '<li class="history-empty">No reviews analyzed yet.</li>';
        return;
    }

    historyList.innerHTML = '';
    history.forEach((item, index) => {
        const li = document.createElement('li');
        li.className = 'history-item';
        
        const lrTag = item.lr.sentiment.substring(0, 3).toLowerCase();
        const lstmTag = item.lstm.sentiment.substring(0, 3).toLowerCase();

        li.innerHTML = `
            <div class="history-text" title="${item.text}">${escapeHtml(item.text)}</div>
            <div class="history-footer">
                <span class="history-time">${formatTimeAgo(item.timestamp)}</span>
                <div class="history-sentiments">
                    <span>LR: <span class="history-tag ${lrTag}">${item.lr.sentiment}</span></span>
                    <span>LSTM: <span class="history-tag ${lstmTag}">${item.lstm.sentiment}</span></span>
                </div>
            </div>
        `;

        li.addEventListener('click', () => {
            document.getElementById('review-input').value = item.text;
            document.getElementById('char-current').textContent = item.text.length.toLocaleString();
            renderResults(item);
            
            // Switch back to Single Analysis tab if on another
            const singleTabBtn = document.querySelector('[data-tab="single-analysis"]');
            if (singleTabBtn) singleTabBtn.click();
        });

        historyList.appendChild(li);
    });
}

function saveToHistory(predictionItem) {
    const history = JSON.parse(localStorage.getItem('sentifilm_history') || '[]');
    
    // Add timestamp
    predictionItem.timestamp = Date.now();
    
    // Remove if review text is already present (move to top)
    const filteredHistory = history.filter(item => item.text !== predictionItem.text);
    filteredHistory.unshift(predictionItem);

    // Keep last 5 entries
    const truncatedHistory = filteredHistory.slice(0, 5);
    localStorage.setItem('sentifilm_history', JSON.stringify(truncatedHistory));
    
    loadHistory();
}

// 7. Toast Alerts and General Helpers
function showToast(message) {
    const toast = document.getElementById('toast');
    const toastMsg = document.getElementById('toast-message');

    toastMsg.textContent = message;
    toast.classList.remove('hidden');
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.classList.add('hidden'), 300);
    }, 4000);
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatTimeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
}
