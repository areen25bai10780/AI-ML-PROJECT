document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dtStatus = document.getElementById('dataset-status');
    const mdStatus = document.getElementById('model-status');
    const initPanel = document.getElementById('init-panel');
    const trainBtn = document.getElementById('train-btn');
    const predictBtn = document.getElementById('predict-btn');
    const headlineInput = document.getElementById('headline-input');
    const resultsPanel = document.getElementById('results-panel');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    // Check Status on Load
    checkStatus();

    // Event Listeners
    trainBtn.addEventListener('click', handleTraining);
    predictBtn.addEventListener('click', handlePrediction);
    
    headlineInput.addEventListener('input', (e) => {
        const val = e.target.value.trim();
        // Only enable if models are ready and text is long enough
        if (mdStatus.classList.contains('active') && val.length >= 5) {
            predictBtn.disabled = false;
        } else {
            predictBtn.disabled = true;
            resultsPanel.style.display = 'none';
        }
    });

    // Functions
    async function checkStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            
            updateStatusUI(data);
        } catch (e) {
            showToast('Failed to connect to server.', true);
        }
    }

    function updateStatusUI(status) {
        if (status.dataset_exists) {
            dtStatus.classList.add('active');
        } else {
            dtStatus.classList.remove('active');
        }

        if (status.models_exist) {
            mdStatus.classList.add('active');
            initPanel.style.display = 'none';
            if (headlineInput.value.trim().length >= 5) {
                predictBtn.disabled = false;
            }
        } else {
            mdStatus.classList.remove('active');
            initPanel.style.display = 'block';
            predictBtn.disabled = true;
        }
    }

    async function handleTraining() {
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing...';
        
        try {
            const res = await fetch('/api/train', { method: 'POST' });
            const data = await res.json();
            
            if (res.ok) {
                showToast(`Success! LR: ${data.details.lr_accuracy}%, DT: ${data.details.dt_accuracy}%`);
                checkStatus();
            } else {
                showToast(data.message || 'Training failed', true);
            }
        } catch (e) {
            showToast('Failed to initialize models.', true);
        } finally {
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="fas fa-cogs"></i> Initialize Models';
        }
    }

    async function handlePrediction() {
        const text = headlineInput.value.trim();
        if (text.length < 5) return;

        // UI Prep
        predictBtn.disabled = true;
        resultsPanel.style.display = 'none';
        loadingSpinner.style.display = 'block';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();

            if (res.ok) {
                displayResults(data.results);
            } else {
                showToast(data.message, true);
            }
        } catch (e) {
            showToast('Prediction failed. Is the server running?', true);
        } finally {
            predictBtn.disabled = false;
            loadingSpinner.style.display = 'none';
        }
    }

    function displayResults(results) {
        resultsPanel.style.display = 'block';

        // Final Verdict Banner
        const banner = document.getElementById('final-verdict-banner');
        const verdictText = document.getElementById('verdict-text');
        
        banner.className = `verdict-banner ${results.final_verdict.toLowerCase()}`;
        verdictText.innerHTML = results.final_verdict === 'REAL' ? 
            '<i class="fas fa-check-circle"></i> VERDICT: REAL NEWS' : 
            '<i class="fas fa-times-circle"></i> VERDICT: FAKE NEWS';

        // Update Models Cards
        updateModelCard('lr', results.logistic_regression);
        updateModelCard('dt', results.decision_tree);
    }

    function updateModelCard(prefix, result) {
        const type = result.prediction.toLowerCase();
        
        const valEl = document.getElementById(`${prefix}-pred`);
        valEl.textContent = result.prediction;
        valEl.className = `prediction-value ${type}`;

        const fillEl = document.getElementById(`${prefix}-conf-fill`);
        // Slight delay for animation effect
        setTimeout(() => {
            fillEl.style.width = `${result.confidence}%`;
            fillEl.className = `fill ${type}`;
        }, 50);

        document.getElementById(`${prefix}-conf-text`).textContent = `${result.confidence}% Confidence`;
    }

    function showToast(message, isError = false) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        if (isError) toast.style.background = 'var(--danger)';
        toast.textContent = message;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
});
