// BarkVision Client Application Logic

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const previewWrapper = document.getElementById('preview-wrapper');
  const previewImg = document.getElementById('preview-img');
  const previewFilename = document.getElementById('preview-filename');
  const removeImgBtn = document.getElementById('remove-img-btn');
  const predictBtn = document.getElementById('predict-btn');
  const predictBtnText = document.getElementById('predict-btn-text');
  const btnSpinner = document.getElementById('btn-spinner');
  const sampleChips = document.querySelectorAll('.sample-chip');

  const emptyState = document.getElementById('empty-state');
  const loadingState = document.getElementById('loading-state');
  const resultsContent = document.getElementById('results-content');

  const resSpeciesName = document.getElementById('res-species-name');
  const resCommonName = document.getElementById('res-common-name');
  const resConfidenceVal = document.getElementById('res-confidence-val');
  const top3Bars = document.getElementById('top3-bars');
  const campusCountDesc = document.getElementById('campus-count-desc');
  const campusGrid = document.getElementById('campus-grid');
  const locationFilter = document.getElementById('location-filter');

  let currentFile = null;
  let currentCampusRecords = [];

  // 1. File Upload & Drag-and-Drop
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', () => {
    if (!currentFile) {
      fileInput.click();
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      handleSelectedFile(fileInput.files[0]);
    }
  });

  removeImgBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/') && !file.name.toLowerCase().endsWith('.webp')) {
      alert('Please upload a valid image file (JPEG, PNG, WEBP).');
      return;
    }
    currentFile = file;
    previewFilename.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzonePrompt.style.display = 'none';
      previewWrapper.style.display = 'block';
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dropzonePrompt.style.display = 'block';
    previewWrapper.style.display = 'none';
    predictBtn.disabled = true;
    showEmptyState();
  }

  // 2. Sample Chips Picker
  sampleChips.forEach((chip) => {
    chip.addEventListener('click', async () => {
      const filename = chip.getAttribute('data-file');
      try {
        chip.style.opacity = '0.6';
        const response = await fetch(`/samples/${encodeURIComponent(filename)}`);
        if (!response.ok) throw new Error('Failed to load sample image');
        const blob = await response.blob();
        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
        handleSelectedFile(file);
        chip.style.opacity = '1';

        // Auto trigger prediction for samples for quick demo!
        setTimeout(() => runPrediction(), 200);
      } catch (err) {
        console.error(err);
        chip.style.opacity = '1';
        alert(`Could not load test sample '${filename}'.`);
      }
    });
  });

  // 3. Prediction Action
  predictBtn.addEventListener('click', () => {
    runPrediction();
  });

  async function runPrediction() {
    if (!currentFile) return;

    setLoading(true);

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with status ${response.status}`);
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      console.error('Prediction error:', err);
      alert(`Prediction failed: ${err.message}`);
      showEmptyState();
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      predictBtn.disabled = true;
      predictBtnText.textContent = 'Analyzing Bark...';
      btnSpinner.style.display = 'inline-block';
      emptyState.style.display = 'none';
      resultsContent.style.display = 'none';
      loadingState.style.display = 'flex';
    } else {
      predictBtn.disabled = !currentFile;
      predictBtnText.textContent = 'Classify & Lookup Campus Data';
      btnSpinner.style.display = 'none';
      loadingState.style.display = 'none';
    }
  }

  function showEmptyState() {
    emptyState.style.display = 'flex';
    loadingState.style.display = 'none';
    resultsContent.style.display = 'none';
  }

  // 4. Render Prediction & Campus Records
  function renderResults(data) {
    emptyState.style.display = 'none';
    loadingState.style.display = 'none';
    resultsContent.style.display = 'block';

    // Hero Card
    resSpeciesName.textContent = data.species_name;
    resCommonName.textContent = data.common_name || 'Campus Tree';
    resConfidenceVal.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // Top 3 Candidates Breakdown
    top3Bars.innerHTML = '';
    if (data.top3 && data.top3.length > 0) {
      data.top3.forEach((item) => {
        const pct = (item.probability * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'top3-item';
        div.innerHTML = `
          <div class="top3-label-row">
            <span><strong>${item.name}</strong></span>
            <span>${pct}%</span>
          </div>
          <div class="top3-bar-bg">
            <div class="top3-bar-fill" style="width: ${pct}%"></div>
          </div>
        `;
        top3Bars.appendChild(div);
      });
    }

    // Campus Records
    currentCampusRecords = data.campus_records || [];
    campusCountDesc.textContent = `${currentCampusRecords.length} trees recorded on campus database`;
    renderCampusGrid(currentCampusRecords);
  }

  function renderCampusGrid(records) {
    campusGrid.innerHTML = '';
    if (!records || records.length === 0) {
      campusGrid.innerHTML = `
        <div class="no-records">
          No specific location records registered for this species in Campus_Trees.json.
        </div>
      `;
      return;
    }

    records.forEach((rec, idx) => {
      const card = document.createElement('div');
      card.className = 'campus-card';
      const loc = rec.location || 'Campus Grounds';
      const planter = rec.planter || 'Campus Authority';
      const desig = rec.designation || '';

      card.innerHTML = `
        <div>
          <div class="location-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
              <circle cx="12" cy="10" r="3"/>
            </svg>
            <span>${escapeHtml(loc)}</span>
          </div>
          <div class="planter-info">
            <span class="planter-label">Planted By</span>
            <div class="planter-name">${escapeHtml(planter)}</div>
            ${desig ? `<div class="planter-desig">${escapeHtml(desig)}</div>` : ''}
          </div>
        </div>
      `;
      campusGrid.appendChild(card);
    });
  }

  // 5. Search / Filter Campus Records
  locationFilter.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    if (!query) {
      renderCampusGrid(currentCampusRecords);
      return;
    }
    const filtered = currentCampusRecords.filter((rec) => {
      const loc = (rec.location || '').toLowerCase();
      const planter = (rec.planter || '').toLowerCase();
      const desig = (rec.designation || '').toLowerCase();
      return loc.includes(query) || planter.includes(query) || desig.includes(query);
    });
    renderCampusGrid(filtered);
  });

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
});
