/* MaizeScan — Main JavaScript */

// Mobile nav toggle
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('navToggle');
    const mobile = document.getElementById('navMobile');
    if (toggle && mobile) {
      toggle.addEventListener('click', () => mobile.classList.toggle('open'));
    }
  });
  