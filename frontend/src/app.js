import { useState } from "react";

function App() {
    const [inputData, setInputData] = useState("");
    const [result, setResult] = useState("");

    const API_URL = "https://bank-fraud-detection-project-1.onrender.com"; // 👈 PUT YOUR RENDER LINK HERE

    const handleSubmit = async () => {
        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    features: inputData.split(",").map(Number)
                })
            });

            const data = await response.json();
            setResult(data.prediction);
        } catch (error) {
            console.error(error);
            setResult("Error connecting to backend");
        }
    };

    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h1>Bank Fraud Detection</h1>

            <input
                type="text"
                placeholder="Enter values (comma separated)"
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                style={{ padding: "10px", width: "300px" }}
            />

            <br /><br />

            <button onClick={handleSubmit} style={{ padding: "10px 20px" }}>
                Predict
            </button>

            <h2>Result: {result}</h2>
        </div>
    );
}

export default App;
