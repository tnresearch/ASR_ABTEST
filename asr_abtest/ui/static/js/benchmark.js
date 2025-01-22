class BenchmarkManager {
    constructor() {
        this.files = new Map();
        this.initializeListeners();
    }

    initializeListeners() {
        document.getElementById('batch-files').addEventListener('change', this.handleFileSelect.bind(this));
        document.getElementById('start-benchmark').addEventListener('click', this.startBenchmark.bind(this));
    }

    async handleFileSelect(event) {
        const files = event.target.files;
        for (let file of files) {
            if (file.name.endsWith('.wav')) {
                this.files.set(file.name, {
                    audio: file,
                    reference: null
                });
            } else if (file.name.endsWith('.json')) {
                const baseName = file.name.replace('.json', '.wav');
                if (this.files.has(baseName)) {
                    this.files.get(baseName).reference = file;
                }
            }
        }
        this.updateFileList();
    }

    async startBenchmark() {
        const progressSection = document.querySelector('.progress-section');
        progressSection.style.display = 'block';

        const formData = new FormData();
        for (let [name, files] of this.files) {
            formData.append('audio_files[]', files.audio);
            formData.append('reference_files[]', files.reference);
        }

        const modelId = document.getElementById('model-select').value;
        formData.append('model_id', modelId);

        try {
            const response = await fetch('/benchmark/process', {
                method: 'POST',
                body: formData
            });

            const results = await response.json();
            this.displayResults(results);
        } catch (error) {
            console.error('Benchmark failed:', error);
            alert('Benchmark failed. Please check console for details.');
        }
    }

    displayResults(results) {
        document.querySelector('.results-section').style.display = 'block';
        document.getElementById('avg-wer').textContent = 
            (results.aggregated_metrics.wer * 100).toFixed(2) + '%';
        document.getElementById('files-processed').textContent = 
            results.num_files;
        // ... populate other results
    }
} 