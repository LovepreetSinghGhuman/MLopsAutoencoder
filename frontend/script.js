document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('upload-form');
  const fileInput = document.getElementById('file-input');
  const resultsDiv = document.getElementById('results');
  const loadingDiv = document.getElementById('loading');

  function renderResult(result) {
    // Force column order if keys exist
    if (result.TransactionID && result.IsFraud) {
      let html = '<div style="overflow-x:auto; max-height:400px; overflow-y:auto;"><table border="1" style="border-collapse:collapse;width:100%">';
      html += "<tr><th>TransactionID</th><th>IsFraud</th></tr>";
      for (let i = 0; i < result.TransactionID.length; i++) {
        html += `<tr><td>${result.TransactionID[i]}</td><td>${result.IsFraud[i]}</td></tr>`;
      }
      html += "</table></div>";
      return html;
    }
    // fallback to previous logic
    if (typeof result === "object" && !Array.isArray(result)) {
      let keys = Object.keys(result);
      let length = result[keys[0]].length;
      let html = '<div style="overflow-x:auto; max-height:400px; overflow-y:auto;"><table border="1" style="border-collapse:collapse;width:100%">';
      html += "<tr>" + keys.map(k => `<th>${k}</th>`).join("") + "</tr>";
      for (let i = 0; i < length; i++) {
        html += "<tr>" + keys.map(k => `<td>${result[k][i]}</td>`).join("") + "</tr>";
      }
      html += "</table></div>";
      return html;
    }
    return `<pre>${JSON.stringify(result, null, 2)}</pre>`;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

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
      resultsDiv.innerHTML = renderResult(result);
    } catch (err) {
      resultsDiv.innerHTML = `<span style="color:red;">Upload failed: ${err.message}</span>`;
    } finally {
      loadingDiv.style.display = "none";
    }
  });
});