document.addEventListener('DOMContentLoaded', () => {
    const newsText = document.getElementById('newsText');
    const charCount = document.getElementById('charCount');
    const predictionForm = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnIcon = document.getElementById('btnIcon');
    const resultCard = document.getElementById('resultCard');

    // Result card elements
    const verdictText = document.getElementById('verdictText');
    const probabilityText = document.getElementById('probabilityText');
    const probabilityBar = document.getElementById('probabilityBar');
    const credibilityStatus = document.getElementById('credibilityStatus');
    const gaugePointer = document.getElementById('gaugePointer');
    const gaugePointerVal = document.getElementById('gaugePointerVal');
    const verdictContainer = document.querySelector('.verdict-container');

    // Count characters as user types
    if (newsText && charCount) {
        newsText.addEventListener('input', () => {
            const count = newsText.value.length;
            charCount.textContent = `${count.toLocaleString()} character${count !== 1 ? 's' : ''}`;
        });
    }

    // Handle form submit
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const textValue = newsText.value.strip ? newsText.value.strip() : newsText.value.trim();
            if (!textValue) return;

            // Update button state (Loading state)
            submitBtn.disabled = true;
            btnSpinner.classList.remove('d-none');
            btnIcon.classList.add('d-none');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: textValue })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    // Update Verdict badge text and style class
                    verdictText.textContent = data.prediction.toUpperCase();
                    
                    // Reset verdict classes
                    verdictContainer.className = 'verdict-container p-4 rounded-4 mb-4 text-center';
                    if (data.prediction === 'Real News') {
                        verdictContainer.classList.add('verdict-real');
                    } else {
                        verdictContainer.classList.add('verdict-fake');
                    }

                    // Update Confidence Score
                    probabilityText.textContent = `${data.probability}%`;
                    probabilityBar.style.width = `${data.probability}%`;
                    
                    // Reset confidence progress color
                    probabilityBar.className = 'progress-bar';
                    if (data.prediction === 'Real News') {
                        probabilityBar.classList.add('bg-success');
                    } else {
                        probabilityBar.classList.add('bg-danger');
                    }

                    // Update Credibility status badge
                    credibilityStatus.textContent = data.credibility_status.toUpperCase();
                    credibilityStatus.className = 'credibility-badge-label fs-3 fw-bold px-4 py-2 rounded-pill d-inline-block text-white';
                    
                    if (data.status_color === 'success') {
                        credibilityStatus.classList.add('bg-success');
                    } else if (data.status_color === 'warning') {
                        credibilityStatus.classList.add('bg-warning', 'text-dark');
                    } else {
                        credibilityStatus.classList.add('bg-danger');
                    }

                    // Move Gauge indicator needle
                    gaugePointer.style.left = `${data.credibility}%`;
                    gaugePointerVal.textContent = `${data.credibility}%`;

                    // Show Results card
                    resultCard.classList.remove('d-none');
                    
                    // Scroll to results
                    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    alert(`Error: ${data.error || 'Failed to complete text analysis.'}`);
                }
            } catch (err) {
                console.error(err);
                alert('An error occurred during prediction. Please make sure the backend is active.');
            } finally {
                // Restore button state
                submitBtn.disabled = false;
                btnSpinner.classList.add('d-none');
                btnIcon.classList.remove('d-none');
            }
        });
    }
});
