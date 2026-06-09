// ── CSRF ──
function getCsrf() {
  const el = document.querySelector('[name=csrfmiddlewaretoken]');
  if (el) return el.value;
  const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
  return cookie ? cookie.trim().split('=')[1] : '';
}

// ── Modal helpers ──
function openModal(id) {
  const el = document.getElementById(id);
  if (el) { el.style.display = 'flex'; document.body.style.overflow = 'hidden'; }
}
function closeModal(id) {
  const el = document.getElementById(id);
  if (el) { el.style.display = 'none'; document.body.style.overflow = ''; }
}

// Close modal on overlay click
document.addEventListener('click', function(e) {
  if (e.target.classList.contains('modal-overlay')) {
    e.target.style.display = 'none';
    document.body.style.overflow = '';
  }
});

// Close on Escape
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay').forEach(el => {
      if (el.style.display !== 'none') {
        el.style.display = 'none';
        document.body.style.overflow = '';
      }
    });
  }
});

// ── Sidebar Toggle ──
document.addEventListener('DOMContentLoaded', function() {
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('sidebarToggle');
  if (!sidebar || !toggleBtn) return;

  const key = 'wk_sidebar_collapsed';
  if (localStorage.getItem(key) === '1') sidebar.classList.add('collapsed');

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    localStorage.setItem(key, sidebar.classList.contains('collapsed') ? '1' : '0');
  });
});

// ── Drag & Drop for file zones ──
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.file-upload-zone').forEach(zone => {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = '#4F46E5'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.style.borderColor = '';
      const input = zone.querySelector('input[type=file]');
      if (input && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        zone.querySelector('.upload-text').textContent = e.dataTransfer.files[0].name;
        zone.classList.add('has-file');
      }
    });
  });
});
