const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.getElementById('loading');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  // Show loading animation/text
  loadingDiv.style.display = "block";
  resultsDiv.innerText = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Error: ${err}`);
    }

    const result = await response.json();
    resultsDiv.innerText = JSON.stringify(result, null, 2);
  } catch (err) {
    resultsDiv.innerText = `Upload failed: ${err.message}`;
  } finally {
    // Hide loading animation/text
    loadingDiv.style.display = "none";
  }
});