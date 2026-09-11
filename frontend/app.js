class BoneAgeAnalyzer {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.previewSection = document.getElementById('previewSection');
        this.previewImage = document.getElementById('previewImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.ageResult = document.getElementById('ageResult');
        this.ageMonths = document.getElementById('ageMonths');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceText = document.getElementById('confidenceText');
        this.uncertaintyResult = document.getElementById('uncertaintyResult');
        this.attentionHeatmap = document.getElementById('attentionHeatmap');
        this.modelInfo = document.getElementById('modelInfo');
        this.newAnalysisBtn = document.getElementById('newAnalysisBtn');
        this.retryBtn = document.getElementById('retryBtn');
        this.errorMessage = document.getElementById('errorMessage');
        this.originalXray = document.getElementById('originalXray');
        this.overlayImage = document.getElementById('overlayImage');
    }

    bindEvents() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Button events
        this.analyzeBtn.addEventListener('click', this.analyzeImage.bind(this));
        this.newAnalysisBtn.addEventListener('click', this.resetAnalysis.bind(this));
        this.retryBtn.addEventListener('click', this.hideError.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG)');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size too large. Please select an image under 10MB.');
            return;
        }

        // Store file for analysis
        this.currentFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.showSection('previewSection');
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        if (!this.currentFile) {
            this.showError('No image selected for analysis');
            return;
        }

        this.showSection('loadingSection');

        const formData = new FormData();
        formData.append('file', this.currentFile);

        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const result = await response.json();
            console.log("FULL API RESPONSE:", result);
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'Failed to analyze image. Please check your connection and try again.');
        }
    }

    displayResults(result) {
    console.log("API Result:", result);

    const data = result?.results?.[0];

    if (!data || data.predicted_age_months === undefined) {
        this.showError("Invalid response from server");
        return;
    }

    // Age
    this.ageResult.textContent = data.predicted_age_formatted || "N/A";
    
    this.ageMonths.textContent = data.predicted_age_months
        ? `${Number(data.predicted_age_months).toFixed(1)} months`
        : "N/A";

    // Confidence
    const confidence = result?.confidence_score
        ? (result.confidence_score * 100).toFixed(1)
        : 0;

    this.confidenceFill.style.width = `${confidence}%`;
    this.confidenceText.textContent = `${confidence}%`;

    // Uncertainty
    this.uncertaintyResult.textContent = result?.uncertainty_months
        ? `± ${result.uncertainty_months.toFixed(1)} months`
        : "N/A";

    // Original X-ray
    if (result?.original_xray) {
        this.originalXray.src = result.original_xray;
    }

    // Heatmap
    if (result?.attention_heatmap) {
        this.attentionHeatmap.src = result.attention_heatmap;
    }

    // Model info
    if (result?.model_info) {
        this.modelInfo.innerHTML = `
            <p><strong>Backbone:</strong> ${result.model_info.backbone}</p>
            <p><strong>Attention Type:</strong> ${result.model_info.attention_type}</p>
            <p><strong>Device:</strong> ${result.model_info.device}</p>
        `;
    }

    this.showSection('resultsSection');
}

        showSection(sectionId) {
        // Hide all sections
        const sections = ['previewSection', 'loadingSection', 'resultsSection', 'errorSection'];
        sections.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });

        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.style.display = 'block';
        }
    }

        showError(message) {
        this.errorMessage.textContent = message;
        this.showSection('errorSection');
        }

        hideError() {
        this.showSection('previewSection');
        }

        resetAnalysis() {
        // Clear file input
        this.fileInput.value = '';
        this.currentFile = null;

        // Reset preview
        this.previewImage.src = '';

        // Hide all sections except upload
        const sections = ['previewSection', 'loadingSection', 'resultsSection', 'errorSection'];
        sections.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
    }

    // Utility method to check API health
    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const health = await response.json();
            console.log('API Health:', health);
            return health.status === 'healthy';
        } catch (error) {
            console.error('API health check failed:', error);
            return false;
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const analyzer = new BoneAgeAnalyzer();
    
    // Optional: Check API health on startup
    analyzer.checkApiHealth().then(healthy => {
        if (!healthy) {
            console.warn('API may not be available. Please ensure the backend server is running.');
        }
    });
});