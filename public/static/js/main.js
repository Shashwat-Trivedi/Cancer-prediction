/**
 * Breast Cancer Prediction System - Client Script
 * Provides:
 * 1. 1-Click sample case loading (Benign vs Malignant demo data)
 * 2. Continuous fresh test values generator (every click fills with a new real case!)
 * 3. Form input validation (ensuring valid numeric entries for all 30 features)
 * 4. Feature grouping tab filters (Mean, SE, Worst)
 * 5. Micro-animations for confidence meter
 */

// Fallback sample values if DOM script tag is absent
const DEFAULT_SAMPLES = {
  benign: {
    radius_mean: 13.54, texture_mean: 14.36, perimeter_mean: 87.46, area_mean: 566.3,
    smoothness_mean: 0.0978, compactness_mean: 0.0813, concavity_mean: 0.0666, concave_points_mean: 0.0478,
    symmetry_mean: 0.1885, fractal_dimension_mean: 0.0577, radius_se: 0.2699, texture_se: 0.7886,
    perimeter_se: 2.058, area_se: 23.56, smoothness_se: 0.0085, compactness_se: 0.0146,
    concavity_se: 0.0239, concave_points_se: 0.0132, symmetry_se: 0.0198, fractal_dimension_se: 0.0023,
    radius_worst: 15.11, texture_worst: 19.26, perimeter_worst: 99.7, area_worst: 711.2,
    smoothness_worst: 0.144, compactness_worst: 0.1773, concavity_worst: 0.239,
    concave_points_worst: 0.1288, symmetry_worst: 0.2977, fractal_dimension_worst: 0.0726
  },
  malignant: {
    radius_mean: 17.99, texture_mean: 10.38, perimeter_mean: 122.8, area_mean: 1001.0,
    smoothness_mean: 0.1184, compactness_mean: 0.2776, concavity_mean: 0.3001, concave_points_mean: 0.1471,
    symmetry_mean: 0.2419, fractal_dimension_mean: 0.0787, radius_se: 1.095, texture_se: 0.9053,
    perimeter_se: 8.589, area_se: 153.4, smoothness_se: 0.0064, compactness_se: 0.049,
    concavity_se: 0.0537, concave_points_se: 0.0159, symmetry_se: 0.03, fractal_dimension_se: 0.0062,
    radius_worst: 25.38, texture_worst: 17.33, perimeter_worst: 184.6, area_worst: 2019.0,
    smoothness_worst: 0.1622, compactness_worst: 0.6656, concavity_worst: 0.7119,
    concave_points_worst: 0.2654, symmetry_worst: 0.4601, fractal_dimension_worst: 0.1189
  }
};

// Track current test sample index so EVERY click loads a completely NEW case
let currentSampleIndex = -1;

/**
 * Retrieve standard two demo cases
 */
function getSampleData() {
  const scriptTag = document.getElementById('sample-data-json');
  if (scriptTag && scriptTag.textContent.trim()) {
    try {
      return JSON.parse(scriptTag.textContent);
    } catch (e) {
      console.warn('Could not parse embedded sample data, using defaults', e);
    }
  }
  return DEFAULT_SAMPLES;
}

/**
 * Retrieve the full array of authentic test cases from Wisconsin dataset
 */
function getTestSamples() {
  const scriptTag = document.getElementById('test-samples-json');
  if (scriptTag && scriptTag.textContent.trim()) {
    try {
      return JSON.parse(scriptTag.textContent);
    } catch (e) {
      console.warn('Could not parse test-samples-json', e);
    }
  }
  return [];
}

/**
 * Fill form fields with values and trigger visual feedback
 * @param {Object} dataObj - Dictionary of feature values
 * @param {'benign'|'malignant'} sampleType - Class type for styling
 */
function applyValuesToForm(dataObj, sampleType) {
  clearValidationErrors();

  for (const [key, value] of Object.entries(dataObj)) {
    const input = document.getElementById(`input-${key}`);
    if (input) {
      input.value = value;

      // Subtle flash feedback on field group
      const group = document.getElementById(`group-${key}`);
      if (group) {
        group.style.transition = 'background-color 0.4s ease, border-color 0.4s ease';
        group.style.backgroundColor = sampleType === 'benign' ? '#f0fdf4' : '#fef2f2';
        group.style.borderColor = sampleType === 'benign' ? '#86efac' : '#fca5a5';
        setTimeout(() => {
          group.style.backgroundColor = '';
          group.style.borderColor = '';
        }, 550);
      }
    }
  }
}

/**
 * Fill form fields with either benign or malignant test data
 * @param {'benign'|'malignant'} sampleType
 */
function fillSampleData(sampleType) {
  const samples = getSampleData();
  const data = samples[sampleType];

  if (!data) {
    console.error(`Unknown sample case: ${sampleType}`);
    return;
  }

  applyValuesToForm(data, sampleType);

  // Update dynamic banner
  showSampleBanner(
    sampleType === 'benign' ? 'Standard Benign Case Loaded' : 'Standard Malignant Case Loaded',
    `Loaded standard ${sampleType.toUpperCase()} test case from Wisconsin dataset. All 30 features ready for prediction!`,
    sampleType
  );

  // Brief button feedback
  const btn = document.getElementById(`btn-fill-${sampleType}`);
  if (btn) {
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span>✓ Loaded ${sampleType.toUpperCase()}</span>`;
    setTimeout(() => {
      btn.innerHTML = originalText;
    }, 1200);
  }
}

/**
 * Fill form with fresh test values on every single click
 * Cycles through 30 real Wisconsin dataset cases (alternating benign & malignant)
 */
function fillRandomTestCase() {
  const testList = getTestSamples();

  let selectedCase = null;
  let caseNumber = 1;
  let totalCases = 30;

  if (testList && testList.length > 0) {
    totalCases = testList.length;
    // Advance to next case index so every click is a NEW set of numbers
    currentSampleIndex = (currentSampleIndex + 1) % totalCases;
    selectedCase = testList[currentSampleIndex];
    caseNumber = currentSampleIndex + 1;
  } else {
    // Fallback if test-samples-json wasn't loaded
    const samples = getSampleData();
    currentSampleIndex = (currentSampleIndex + 1) % 2;
    const type = currentSampleIndex === 0 ? 'benign' : 'malignant';
    selectedCase = {
      id: `Demo Case #${currentSampleIndex + 1}`,
      type: type,
      label: type === 'benign' ? 'Benign Sample' : 'Malignant Sample',
      data: samples[type]
    };
  }

  // Populate the 30 input fields
  applyValuesToForm(selectedCase.data, selectedCase.type);

  // Update badge on button
  const badge = document.getElementById('sample-case-badge');
  if (badge) {
    const typeLabel = selectedCase.type.toUpperCase();
    badge.textContent = `Case ${caseNumber}/${totalCases} (${typeLabel})`;
    badge.className = `sample-counter-pill ${selectedCase.type}`;
  }

  // Show dynamic notification banner
  const typeText = selectedCase.type === 'benign' ? 'Benign (Healthy Profile)' : 'Malignant (High-Risk Profile)';
  showSampleBanner(
    `Loaded ${selectedCase.id} — ${typeText}`,
    `All 30 numeric biomarkers populated with fresh test measurements! (Click button again anytime to test Case #${(caseNumber % totalCases) + 1})`,
    selectedCase.type
  );

  // Subtle rotation animation on dice icon
  const diceIcons = document.querySelectorAll('.dice-icon');
  diceIcons.forEach(icon => {
    icon.style.transition = 'transform 0.4s ease';
    icon.style.transform = `rotate(${(caseNumber * 90)}deg) scale(1.15)`;
    setTimeout(() => {
      icon.style.transform = '';
    }, 450);
  });
}

/**
 * Display notification banner showing details of the loaded sample
 */
function showSampleBanner(title, description, sampleType) {
  const banner = document.getElementById('sample-status-banner');
  const titleEl = document.getElementById('sample-status-title');
  const descEl = document.getElementById('sample-status-desc');
  const iconEl = document.getElementById('sample-status-icon');

  if (banner && titleEl && descEl) {
    titleEl.textContent = title;
    descEl.textContent = description;
    iconEl.textContent = sampleType === 'benign' ? '🟢' : '🔴';
    banner.style.display = 'flex';
  }
}

/**
 * Dismiss sample notification banner
 */
function closeSampleBanner() {
  const banner = document.getElementById('sample-status-banner');
  if (banner) {
    banner.style.display = 'none';
  }
}

/**
 * Clear all input fields and validation states
 */
function clearAllInputs() {
  clearValidationErrors();
  closeSampleBanner();
  const form = document.getElementById('prediction-form');
  if (form) {
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
      input.value = '';
    });
  }

  const badge = document.getElementById('sample-case-badge');
  if (badge) {
    badge.textContent = 'Cleared';
    badge.className = 'sample-counter-pill';
  }
}

/**
 * Remove invalid field visual highlights
 */
function clearValidationErrors() {
  const invalidGroups = document.querySelectorAll('.feature-input-group.invalid-field');
  invalidGroups.forEach(group => group.classList.remove('invalid-field'));

  const existingAlert = document.getElementById('client-validation-alert');
  if (existingAlert) {
    existingAlert.remove();
  }
}

/**
 * Category Tab Filter (All, Mean, SE, Worst)
 * Shows/hides inputs matching the selected biomarker category
 */
function filterCategory(categoryName, tabButton) {
  // Update active tab button
  const allTabs = document.querySelectorAll('.cat-tab');
  allTabs.forEach(tab => tab.classList.remove('active'));
  if (tabButton) {
    tabButton.classList.add('active');
  }

  // Show or hide feature inputs
  const allGroups = document.querySelectorAll('.feature-input-group');
  allGroups.forEach(group => {
    const groupCategory = group.getAttribute('data-group');
    if (categoryName === 'all' || groupCategory === categoryName) {
      group.style.display = '';
    } else {
      group.style.display = 'none';
    }
  });
}

/**
 * Validate that every single one of the 30 features contains a valid number
 */
function validateForm(event) {
  clearValidationErrors();
  const form = document.getElementById('prediction-form');
  if (!form) return true;

  const inputs = form.querySelectorAll('input[type="number"]');
  let firstInvalid = null;
  let emptyCount = 0;

  inputs.forEach(input => {
    const val = input.value.trim();
    const num = parseFloat(val);

    if (val === '' || isNaN(num)) {
      emptyCount++;
      const parentGroup = input.closest('.feature-input-group');
      if (parentGroup) {
        parentGroup.classList.add('invalid-field');
      }
      if (!firstInvalid) {
        firstInvalid = input;
      }
    }
  });

  if (emptyCount > 0) {
    event.preventDefault();

    // If inputs are currently hidden by a category tab, reset tab to "All"
    filterCategory('all', document.querySelector('.cat-tab[data-target="all"]'));

    // Create a friendly client-side alert
    const formCard = document.querySelector('.form-container-card');
    const alertDiv = document.createElement('div');
    alertDiv.id = 'client-validation-alert';
    alertDiv.className = 'alert alert-danger';
    alertDiv.innerHTML = `
      <div class="alert-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      </div>
      <div class="alert-text">
        <strong>Incomplete Form:</strong> Please enter valid numeric values for all 30 features (${emptyCount} missing or invalid). 
        You can simply click <strong>"🎲 Fill New Test Values"</strong> above to auto-populate the form instantly!
      </div>
    `;

    if (formCard) {
      formCard.insertBefore(alertDiv, formCard.firstChild);
    }

    // Scroll to the first invalid field
    if (firstInvalid) {
      firstInvalid.focus();
      firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    return false;
  }

  return true;
}

/**
 * On Page Load: Animate confidence gauge if present
 */
document.addEventListener('DOMContentLoaded', () => {
  const gaugeFill = document.querySelector('.gauge-bar-fill');
  if (gaugeFill) {
    const targetWidth = gaugeFill.style.width;
    gaugeFill.style.width = '0%';
    setTimeout(() => {
      gaugeFill.style.width = targetWidth;
    }, 150);
  }
});
