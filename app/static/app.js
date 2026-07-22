const API_BASE = '';
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const POLL_INTERVALS = [500, 1000, 1500, 2000, 3000, 4000, 5000];

let selectedFile = null;
let currentImageId = null;
let pollTimer = null;
let pollIndex = 0;

document.addEventListener('DOMContentLoaded', init);

function init() {
  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Upload zone events
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');

  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  });

  document.getElementById('btn-remove').addEventListener('click', resetUpload);
  document.getElementById('btn-upload').addEventListener('click', uploadImage);
  document.getElementById('btn-new-upload').addEventListener('click', resetUpload);
  document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
  document.getElementById('toast-close').addEventListener('click', () => {
    document.getElementById('error-toast').style.display = 'none';
  });

  // Initial data load
  loadStats();
  loadHistory();
}

function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');

  document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
  document.getElementById(`tab-${tabName}`).style.display = 'block';

  if (tabName === 'dashboard') loadStats();
  if (tabName === 'history') loadHistory();
}

function handleFileSelect(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    showError('Unsupported file type. Please upload JPG, PNG, WebP, or BMP.');
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError('File exceeds 10 MB limit.');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('preview-image').src = e.target.result;
    document.getElementById('result-image').src = e.target.result;
    document.getElementById('preview-name').textContent = file.name;
    document.getElementById('preview-size').textContent = formatFileSize(file.size);

    document.getElementById('upload-zone').style.display = 'none';
    document.getElementById('preview-container').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

async function uploadImage() {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append('file', selectedFile);

  document.getElementById('processing-overlay').style.display = 'flex';
  clearProcessingSteps();
  animateStep('step-upload');
  document.getElementById('processing-status').textContent = 'Uploading...';

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (res.status === 413) throw new Error('File too large');
    if (res.status === 415) throw new Error('Unsupported media type');

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');

    if (data.id) {
      startPolling(data.id);
    } else {
      throw new Error('No image ID received');
    }
  } catch (error) {
    document.getElementById('processing-overlay').style.display = 'none';
    showError(error.message);
  }
}

function startPolling(imageId) {
  currentImageId = imageId;
  pollIndex = 0;
  pollStatus();
}

async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/status/${currentImageId}`);
    if (!res.ok) throw new Error('Failed to fetch status');
    const data = await res.json();

    if (data.status === 'COMPLETED') {
      animateStep('step-complete');
      setTimeout(() => fetchResults(currentImageId), 400);
      return;
    } else if (data.status === 'FAILED') {
      document.getElementById('processing-overlay').style.display = 'none';
      showError(`Analysis failed: ${data.error_message || 'Unknown error'}`);
      return;
    } else {
      animateStep('step-blur');
      document.getElementById('processing-status').textContent = 'Processing...';
    }

    const interval = POLL_INTERVALS[Math.min(pollIndex, POLL_INTERVALS.length - 1)];
    pollIndex++;
    pollTimer = setTimeout(pollStatus, interval);
  } catch (error) {
    document.getElementById('processing-overlay').style.display = 'none';
    showError(error.message);
  }
}

async function fetchResults(imageId) {
  try {
    const res = await fetch(`${API_BASE}/result/${imageId}`);
    if (!res.ok) throw new Error('Failed to fetch results');
    const data = await res.json();

    document.getElementById('processing-overlay').style.display = 'none';
    document.getElementById('preview-container').style.display = 'none';

    displayResults(data);
    document.getElementById('result-card').style.display = 'block';
  } catch (error) {
    document.getElementById('processing-overlay').style.display = 'none';
    showError(error.message);
  }
}

function displayResults(data) {
  // Status badge
  const badge = document.getElementById('result-status');
  badge.textContent = data.overall_status.replace('_', ' ');
  badge.className = 'status-badge';
  if (data.overall_status === 'OK') badge.classList.add('status-ok');
  else if (data.overall_status === 'NEEDS_REVIEW') badge.classList.add('status-review');
  else badge.classList.add('status-rejected');

  document.getElementById('result-summary').textContent = data.summary || 'Analysis complete.';

  // Confidence gauge
  const confFill = document.getElementById('confidence-fill');
  const confValue = document.getElementById('confidence-value');
  const pct = (data.confidence_score * 100).toFixed(0);
  confFill.style.width = `${pct}%`;
  confValue.textContent = `${pct}%`;

  if (data.confidence_score >= 0.7) confFill.style.background = 'var(--success)';
  else if (data.confidence_score >= 0.4) confFill.style.background = 'var(--warning)';
  else confFill.style.background = 'var(--danger)';

  // Metrics
  document.getElementById('val-blur').textContent = data.is_blurry ? 'Blurry' : 'Clear';
  document.getElementById('detail-blur').textContent = `Score: ${(data.blur_score || 0).toFixed(2)}`;

  document.getElementById('val-brightness').textContent = data.brightness ? data.brightness.toFixed(2) : 'N/A';
  document.getElementById('val-lowlight').textContent = data.is_low_light ? 'Yes' : 'No';

  document.getElementById('val-duplicate').textContent = data.duplicate ? 'Duplicate found' : 'Original';
  if (data.duplicate && data.duplicate_of) {
    document.getElementById('detail-duplicate').textContent = `Matches: ${data.duplicate_of.substring(0, 8)}`;
  } else {
    document.getElementById('detail-duplicate').textContent = '';
  }

  document.getElementById('val-ocr').textContent = data.ocr_text
    ? (data.ocr_text.length > 50 ? data.ocr_text.substring(0, 50) + '...' : data.ocr_text)
    : 'None detected';

  document.getElementById('val-plate').textContent = data.vehicle_number || 'Not found';
  document.getElementById('detail-plate').textContent = data.plate_valid
    ? 'Valid format'
    : (data.vehicle_number ? 'Invalid format' : '');

  // Metadata
  if (data.image_metadata) {
    const meta = data.image_metadata;
    const sizeStr = meta.file_size_bytes ? formatFileSize(meta.file_size_bytes) : '';
    document.getElementById('detail-metadata').innerHTML = `
      Format: ${meta.format || 'N/A'}<br>
      Dimensions: ${meta.width || '?'} x ${meta.height || '?'}<br>
      Size: ${sizeStr}<br>
      EXIF: ${meta.has_exif ? 'Present' : 'Missing'}
    `;
  } else {
    document.getElementById('detail-metadata').textContent = 'No metadata available';
  }

  document.getElementById('val-screenshot').textContent = data.screenshot ? 'Detected' : 'Not detected';
  document.getElementById('val-tamper').textContent = data.tampered ? 'Potential tampering' : 'None detected';
}

async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) return;
    const data = await res.json();

    const grid = document.getElementById('dashboard-grid');
    grid.innerHTML = `
      <div class="stat-card">
        <h3>${data.total_images || 0}</h3>
        <p>Total Images</p>
      </div>
      <div class="stat-card">
        <h3>${data.completed || 0}</h3>
        <p>Completed</p>
      </div>
      <div class="stat-card">
        <h3>${(data.pending || 0) + (data.processing || 0)}</h3>
        <p>In Progress</p>
      </div>
      <div class="stat-card">
        <h3>${data.failed || 0}</h3>
        <p>Failed</p>
      </div>
      <div class="stat-card">
        <h3>${data.duplicates_detected || 0}</h3>
        <p>Duplicates</p>
      </div>
      <div class="stat-card">
        <h3>${data.average_confidence_score ? (data.average_confidence_score * 100).toFixed(1) : 0}%</h3>
        <p>Avg. Confidence</p>
      </div>
      <div class="stat-card">
        <h3>${data.plate_validation_rate ? (data.plate_validation_rate * 100).toFixed(1) : 0}%</h3>
        <p>Plate Detection</p>
      </div>
    `;
  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

async function loadHistory(limit = 20, offset = 0) {
  try {
    const res = await fetch(`${API_BASE}/images?limit=${limit}&offset=${offset}`);
    if (!res.ok) return;
    const data = await res.json();

    const list = document.getElementById('history-list');
    list.innerHTML = '';

    if (!data.items || data.items.length === 0) {
      list.innerHTML = '<p style="text-align:center; color:var(--text-muted); padding:24px">No uploads yet.</p>';
      return;
    }

    data.items.forEach(item => {
      const el = document.createElement('div');
      el.className = 'history-item';

      let badgeClass = 'status-pending';
      if (item.status === 'COMPLETED') badgeClass = 'status-ok';
      if (item.status === 'FAILED') badgeClass = 'status-failed';
      if (item.status === 'PROCESSING') badgeClass = 'status-processing';

      el.innerHTML = `
        <div class="history-info">
          <span class="history-filename">${item.filename || 'Unknown'}</span>
          <span class="history-date">${formatDate(item.uploaded_at)}</span>
        </div>
        <span class="status-badge ${badgeClass}">${item.status}</span>
        <button class="btn-delete-history" type="button" aria-label="Delete ${item.filename || 'upload'}" title="Delete upload">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/>
          </svg>
          <span>Delete</span>
        </button>
      `;

      el.querySelector('.btn-delete-history').addEventListener('click', async (event) => {
        event.stopPropagation();
        await deleteHistoryItem(item.id, el);
      });

      if (item.status === 'COMPLETED') {
        el.addEventListener('click', async () => {
          switchTab('upload');
          document.getElementById('upload-zone').style.display = 'none';
          document.getElementById('processing-overlay').style.display = 'flex';
          document.getElementById('processing-status').textContent = 'Loading results...';
          await fetchResults(item.id);
        });
      }

      list.appendChild(el);
    });
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

async function deleteHistoryItem(imageId, element) {
  if (!window.confirm('Delete this upload and its analysis?')) return;

  try {
    const res = await fetch(`${API_BASE}/images/${imageId}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not delete the upload');

    element.remove();
    if (!document.querySelector('#history-list .history-item')) loadHistory();
    loadStats();
    showSuccess('Upload and analysis deleted.');
  } catch (error) {
    showError(error.message);
  }
}

async function clearHistory() {
  if (!window.confirm('Delete all uploads and their analyses?')) return;

  try {
    const res = await fetch(`${API_BASE}/images`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not clear history');

    loadHistory();
    loadStats();
    showSuccess(`Deleted ${data.deleted} upload${data.deleted === 1 ? '' : 's'}.`);
  } catch (error) {
    showError(error.message);
  }
}

function showError(message) {
  const toast = document.getElementById('error-toast');
  toast.classList.remove('success-toast');
  document.getElementById('toast-message').textContent = message;
  toast.style.display = 'flex';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 5000);
}

function showSuccess(message) {
  const toast = document.getElementById('error-toast');
  toast.classList.add('success-toast');
  document.getElementById('toast-message').textContent = message;
  toast.style.display = 'flex';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 5000);
}

function resetUpload() {
  selectedFile = null;
  currentImageId = null;
  document.getElementById('file-input').value = '';

  document.getElementById('result-card').style.display = 'none';
  document.getElementById('preview-container').style.display = 'none';
  document.getElementById('upload-zone').style.display = 'flex';
  clearProcessingSteps();
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  return date.toLocaleString();
}

function animateStep(stepId) {
  const step = document.getElementById(stepId);
  if (step) step.classList.add('active');
}

function clearProcessingSteps() {
  document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
}
