/**
 * Auto Frame AI — Client Application
 * End-to-end orchestration for Vehicle Detection, Background Removal,
 * Plate Branding, Studio Compositing, and Cinematic Video Generation.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        selectedFiles: [],      // File objects
        sampleFilenames: [],    // strings from color cars/
        activePreset: 'showroom',
        activeLayout: 'vertical_reel',
        pollInterval: null,
        currentJobId: null,
    };

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const btnLoadDemo = document.getElementById('btn-load-demo');
    const queueSection = document.getElementById('queue-section');
    const queueGrid = document.getElementById('queue-grid');
    const queueCount = document.getElementById('queue-count');
    const btnClearQueue = document.getElementById('btn-clear-queue');
    const btnLaunch = document.getElementById('btn-launch');

    // Form inputs
    const inputDealerName = document.getElementById('input-dealer-name');
    const inputDealerPhone = document.getElementById('input-dealer-phone');
    const inputPrice = document.getElementById('input-price');
    const inputCarTitle = document.getElementById('input-car-title');
    const selectLayout = document.getElementById('select-layout');
    const inputSlideDuration = document.getElementById('input-slide-duration');
    const durationDisplay = document.getElementById('duration-display');

    // Preset cards
    const presetCards = document.querySelectorAll('.preset-card');

    // Progress Section
    const progressSection = document.getElementById('progress-section');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressStage = document.getElementById('progress-stage');
    const progressPct = document.getElementById('progress-pct');
    const consoleMessage = document.getElementById('console-message');

    // Results Section
    const resultsSection = document.getElementById('results-section');
    const videoPlayer = document.getElementById('video-player');
    const btnDownloadVideo = document.getElementById('btn-download-video');
    const specsGrid = document.getElementById('specs-grid');
    const comparisonGrid = document.getElementById('comparison-grid');

    // Stepper items
    const stepItems = document.querySelectorAll('.step-item');

    // ------------------------------------------------------------------------
    // Utility: Show Toast Notifications
    // ------------------------------------------------------------------------
    function showToast(message, type = 'info') {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerText = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ------------------------------------------------------------------------
    // Update Stepper
    // ------------------------------------------------------------------------
    function updateStepper(stepIndex) {
        stepItems.forEach((item, idx) => {
            if (idx < stepIndex) {
                item.classList.add('completed');
                item.classList.remove('active');
            } else if (idx === stepIndex) {
                item.classList.add('active');
                item.classList.remove('completed');
            } else {
                item.classList.remove('active', 'completed');
            }
        });
    }

    // ------------------------------------------------------------------------
    // Drag & Drop File Handling
    // ------------------------------------------------------------------------
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) {
            addFilesToQueue(files);
        }
    });

    btnBrowse.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            addFilesToQueue(files);
        }
        fileInput.value = '';
    });

    function addFilesToQueue(files) {
        files.forEach(file => {
            state.selectedFiles.push(file);
        });
        renderQueue();
    }

    // ------------------------------------------------------------------------
    // Demo Sample Vehicle Loader
    // ------------------------------------------------------------------------
    btnLoadDemo.addEventListener('click', async () => {
        btnLoadDemo.disabled = true;
        btnLoadDemo.innerHTML = '⏳ Loading Demo Car...';

        try {
            const resp = await fetch('/api/samples');
            const data = await resp.json();
            if (data.samples && data.samples.length > 0) {
                // Select 4-5 sample images from the demo set
                const chosen = data.samples.slice(0, 4);
                state.sampleFilenames = chosen.map(s => s.name);
                showToast(`Loaded ${chosen.length} vehicle images from demo collection!`, 'success');
                renderQueueWithSamples(chosen);
            } else {
                showToast('No demo vehicle images available.', 'error');
            }
        } catch (err) {
            console.error(err);
            showToast('Failed to load demo vehicle images.', 'error');
        } finally {
            btnLoadDemo.disabled = false;
            btnLoadDemo.innerHTML = '✨ Load Demo Car';
        }
    });

    function renderQueueWithSamples(samples) {
        queueGrid.innerHTML = '';
        state.selectedFiles = []; // replace with samples for pure demo
        queueSection.style.display = 'block';
        queueCount.innerText = `${samples.length} Image(s)`;

        samples.forEach((sample, idx) => {
            const card = document.createElement('div');
            card.className = 'queue-card';
            card.innerHTML = `
                <img src="${sample.url}" alt="${sample.name}">
                <button class="queue-remove" title="Remove" data-sample-idx="${idx}">✕</button>
                <div class="queue-badge">Demo #${idx + 1}</div>
            `;
            queueGrid.appendChild(card);
        });

        attachQueueListeners();
        btnLaunch.disabled = false;
    }

    // ------------------------------------------------------------------------
    // Render Queue
    // ------------------------------------------------------------------------
    function renderQueue() {
        queueGrid.innerHTML = '';
        const total = state.selectedFiles.length;

        if (total === 0 && state.sampleFilenames.length === 0) {
            queueSection.style.display = 'none';
            btnLaunch.disabled = true;
            return;
        }

        queueSection.style.display = 'block';
        queueCount.innerText = `${total} Image(s)`;

        state.selectedFiles.forEach((file, idx) => {
            const card = document.createElement('div');
            card.className = 'queue-card';

            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = file.name;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'queue-remove';
            removeBtn.innerHTML = '✕';
            removeBtn.title = 'Remove';
            removeBtn.onclick = () => {
                state.selectedFiles.splice(idx, 1);
                renderQueue();
            };

            const badge = document.createElement('div');
            badge.className = 'queue-badge';
            badge.innerText = `#${idx + 1}`;

            card.appendChild(img);
            card.appendChild(removeBtn);
            card.appendChild(badge);
            queueGrid.appendChild(card);
        });

        btnLaunch.disabled = total === 0 && state.sampleFilenames.length === 0;
    }

    function attachQueueListeners() {
        document.querySelectorAll('.queue-remove[data-sample-idx]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.getAttribute('data-sample-idx'), 10);
                state.sampleFilenames.splice(idx, 1);
                if (state.sampleFilenames.length === 0) {
                    queueSection.style.display = 'none';
                    btnLaunch.disabled = true;
                } else {
                    queueCount.innerText = `${state.sampleFilenames.length} Image(s)`;
                    e.target.closest('.queue-card').remove();
                }
            });
        });
    }

    btnClearQueue.addEventListener('click', () => {
        state.selectedFiles = [];
        state.sampleFilenames = [];
        renderQueue();
    });

    // ------------------------------------------------------------------------
    // Preset Card Selection
    // ------------------------------------------------------------------------
    presetCards.forEach(card => {
        card.addEventListener('click', () => {
            presetCards.forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            state.activePreset = card.getAttribute('data-preset');
        });
    });

    // ------------------------------------------------------------------------
    // Slide Duration Slider
    // ------------------------------------------------------------------------
    if (inputSlideDuration) {
        inputSlideDuration.addEventListener('input', (e) => {
            durationDisplay.innerText = `${parseFloat(e.target.value).toFixed(1)}s`;
        });
    }

    // ------------------------------------------------------------------------
    // Launch Full Pipeline
    // ------------------------------------------------------------------------
    btnLaunch.addEventListener('click', async () => {
        if (state.selectedFiles.length === 0 && state.sampleFilenames.length === 0) {
            showToast('Please upload car images or load demo vehicle photos.', 'error');
            return;
        }

        btnLaunch.disabled = true;
        btnLaunch.innerHTML = '⏳ Initializing AI Workers...';

        // Prepare FormData
        const formData = new FormData();
        state.selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        if (state.sampleFilenames.length > 0) {
            formData.append('sample_names', state.sampleFilenames.join(','));
        }

        formData.append('dealer_name', inputDealerName.value.trim() || 'AUTOFRAME MOTORS');
        formData.append('dealer_phone', inputDealerPhone.value.trim() || '+91 98765 43210');
        formData.append('price', inputPrice.value.trim() || '₹ 11,50,000');
        if (inputCarTitle.value.trim()) {
            formData.append('car_title', inputCarTitle.value.trim());
        }
        formData.append('preset', state.activePreset);
        formData.append('layout', selectLayout.value);
        formData.append('slide_duration', inputSlideDuration.value);

        // Show Progress Section
        progressSection.classList.add('visible');
        resultsSection.classList.remove('visible');
        progressBarFill.style.width = '10%';
        progressPct.innerText = '10%';
        progressStage.innerText = '1/5: OCR & Specs Extraction';
        consoleMessage.innerText = 'Extracting vehicle specs and reading license plate with OCR...';
        updateStepper(1);

        try {
            const resp = await fetch('/api/pipeline/start', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || 'Failed to start pipeline');
            }

            const data = await resp.json();
            state.currentJobId = data.job_id;
            showToast(`Pipeline job ${data.job_id} started!`, 'info');

            // Start Polling
            startPollingStatus(data.job_id);

        } catch (err) {
            console.error(err);
            showToast(`Error: ${err.message}`, 'error');
            btnLaunch.disabled = false;
            btnLaunch.innerHTML = '🚀 Run AutoFrame AI Pipeline';
            progressSection.classList.remove('visible');
            updateStepper(0);
        }
    });

    // ------------------------------------------------------------------------
    // Status Polling
    // ------------------------------------------------------------------------
    function startPollingStatus(jobId) {
        if (state.pollInterval) clearInterval(state.pollInterval);

        state.pollInterval = setInterval(async () => {
            try {
                const resp = await fetch(`/api/pipeline/status/${jobId}`);
                if (!resp.ok) return;

                const job = await resp.json();
                const pct = Math.min(100, Math.round((job.progress || 0) * 100));

                progressBarFill.style.width = `${pct}%`;
                progressPct.innerText = `${pct}%`;
                progressStage.innerText = job.current_stage || 'Processing...';
                consoleMessage.innerText = job.message || 'Executing computer vision operations...';

                // Stepper Sync
                if (pct >= 90) updateStepper(4);
                else if (pct >= 70) updateStepper(3);
                else if (pct >= 40) updateStepper(2);
                else if (pct >= 20) updateStepper(1);

                if (job.status === 'completed') {
                    clearInterval(state.pollInterval);
                    onPipelineCompleted(job);
                } else if (job.status === 'failed') {
                    clearInterval(state.pollInterval);
                    onPipelineFailed(job);
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        }, 1200);
    }

    // ------------------------------------------------------------------------
    // Pipeline Completed Handler
    // ------------------------------------------------------------------------
    function onPipelineCompleted(job) {
        progressBarFill.style.width = '100%';
        progressPct.innerText = '100%';
        progressStage.innerText = '🎉 Video Generation Completed!';
        consoleMessage.innerText = 'All images processed, framed into showroom, and video rendered.';
        updateStepper(4);
        showToast('AutoFrame AI Pipeline completed successfully!', 'success');

        btnLaunch.disabled = false;
        btnLaunch.innerHTML = '🚀 Run AutoFrame AI Pipeline';

        // Render Results
        resultsSection.classList.add('visible');

        // Video Player
        if (job.video_url) {
            videoPlayer.src = job.video_url;
            videoPlayer.load();
            btnDownloadVideo.onclick = () => {
                window.location.href = `/api/pipeline/download/${job.job_id}`;
            };
        }

        // Render Specs Chips
        renderSpecsChips(job.specs || {});

        // Render Comparison Carousel Cards
        renderComparisonCards(job.slides_data || []);

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function onPipelineFailed(job) {
        showToast(`Pipeline failed: ${job.error || 'Unknown error'}`, 'error');
        btnLaunch.disabled = false;
        btnLaunch.innerHTML = '🚀 Run AutoFrame AI Pipeline';
        progressStage.innerText = '❌ Pipeline Failed';
        consoleMessage.innerText = job.error || 'An error occurred during pipeline execution.';
    }

    // ------------------------------------------------------------------------
    // Render Extracted Specs Chips
    // ------------------------------------------------------------------------
    function renderSpecsChips(specs) {
        specsGrid.innerHTML = '';
        const fields = [
            { label: 'Registration Number', val: specs.registration_number || 'MH 02 XX 1234' },
            { label: 'Make & Model', val: [specs.make, specs.model, specs.variant].filter(Boolean).join(' ') || 'Hyundai Creta SX(O)' },
            { label: 'Year', val: specs.year || '2022' },
            { label: 'Fuel Type', val: specs.fuel_type || 'Diesel' },
            { label: 'Odometer', val: specs.odometer || '24,500 KM' },
            { label: 'Dealership Brand', val: inputDealerName.value.trim() || 'AUTOFRAME MOTORS' },
        ];

        fields.forEach(f => {
            const chip = document.createElement('div');
            chip.className = 'spec-chip';
            chip.innerHTML = `
                <span>${f.label}</span>
                <strong>${f.val}</strong>
            `;
            specsGrid.appendChild(chip);
        });
    }

    // ------------------------------------------------------------------------
    // Render Before/After Cards
    // ------------------------------------------------------------------------
    function renderComparisonCards(slides) {
        comparisonGrid.innerHTML = '';
        if (!slides || slides.length === 0) return;

        slides.forEach((s, idx) => {
            const card = document.createElement('div');
            card.className = 'comparison-card';

            const viewName = (s.view || 'Exterior').replace('_', ' ').toUpperCase();

            card.innerHTML = `
                <div class="comparison-views">
                    <img src="${s.original_url}" alt="Original #${idx + 1}" title="Original Photo">
                    <img src="${s.framed_url}" alt="Studio Composite #${idx + 1}" title="Framed Studio Showroom">
                </div>
                <div class="comparison-body">
                    <span class="view-tag">${viewName}</span>
                    <span style="font-size:0.75rem; color:var(--text-muted);">Original ⟷ Studio Showroom</span>
                </div>
            `;
            comparisonGrid.appendChild(card);
        });
    }
});
