const model = document.getElementById("model-select").value;

const res = await fetch("/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model: model, prompt: fullPrompt }),
});

