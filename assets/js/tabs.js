function switchTab(tabId) {
  // Hide all tabs
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

  // Show selected tab
  document.getElementById(tabId).classList.add('active');
  event.target.classList.add('active');
}