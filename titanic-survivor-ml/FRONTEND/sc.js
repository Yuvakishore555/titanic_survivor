document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    console.log("Sending to API:", data); // Already inside, correct
    console.log("Request URL:", "/predict"); // Moved inside
    console.log("Request body:", JSON.stringify(data)); // Moved inside

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.prediction) {
        document.getElementById("result").innerText = `✅ Prediction: ${result.prediction}`;
      } else if (result.error) {
        document.getElementById("result").innerText = `❌ Error: ${result.error}`;
      } else {
        document.getElementById("result").innerText = `🤔 Unexpected response: ${JSON.stringify(result)}`;
      }
    } catch (err) {
      console.error("Communication error:", err);
      document.getElementById("result").innerText = "⚠️ Something went wrong.";
    }
  });
});