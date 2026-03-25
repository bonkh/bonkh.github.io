const sliderState = {};

function moveSlide(id, direction) {
  const slider = document.getElementById(id);
  const slides = slider.querySelectorAll('.slide');
  
  if (!sliderState[id]) sliderState[id] = 0;
  
  sliderState[id] = (sliderState[id] + direction + slides.length) % slides.length;
  
  slides.forEach((s, i) => s.classList.toggle('active', i === sliderState[id]));
  updateDots(id, slides.length);
}

function updateDots(id, total) {
  const dotsContainer = document.getElementById(`${id}-dots`);
  dotsContainer.innerHTML = '';
  for (let i = 0; i < total; i++) {
    const dot = document.createElement('span');
    dot.classList.add('dot');
    if (i === sliderState[id]) dot.classList.add('active');
    dot.onclick = () => { sliderState[id] = i - 1; moveSlide(id, 1); };
    dotsContainer.appendChild(dot);
  }
}

// Initialize all sliders on page load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.slider').forEach(s => {
    const slides = s.querySelectorAll('.slide');
    slides.forEach((slide, i) => slide.classList.toggle('active', i === 0));
    updateDots(s.id, slides.length);
  });
});