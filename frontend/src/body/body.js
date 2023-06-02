import React, { useState, useEffect } from "react";
import "./body.css";
import "bootstrap/dist/css/bootstrap.min.css";
function Body() {
  const [outputData, setData] = useState(null);
  const [outputCPData, setCPData] = useState([]);
  const [input_data1, setInputData1] = useState(null);
  const [input_data2, setInputData2] = useState(null);
  const [print_data, setPrint] = useState(false);
  const [show_loading, setLoading] = useState(false);

  const computeSim = async () => {
    setPrint(false);
    setLoading(true);
    const sim = await fetch("http://127.0.0.1:5000/sts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input_data1, input_data2 }),
    })
      .then((res) => res.json())
      .then((data) => {
        setData(data.result);
        setCPData(data.CP);
        setPrint(true);
        console.log(data);
        //console.log(data)
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  function handleInput1Change(event) {
    setInputData1(event.target.value);
  }

  function handleInput2Change(event) {
    setInputData2(event.target.value);
  }

  return (
    <div className="App">
      <h1 className="titleSection"><div className="titleSectionTXT">CP-STS</div></h1>
      <div className="inputSection">
        <input
          type="text"
          className="form-control inputTextArea"
          placeholder="Enter Sentence 1"
          onChange={handleInput1Change}
        />
        <input
          type="text"
          className="form-control inputTextArea"
          placeholder="Enter Sentence 2"
          onChange={handleInput2Change}
        />
      </div>

      <button
        type="submit"
        className="btn btn-dark buttonsection"
        onClick={computeSim}
      >
        Compute
      </button>
      {/* <div className="line"></div> */}
      <div className="card">
        <div className="card-body">
          <h5 className="card-title">Output</h5>
          <div className="output">
            {show_loading && !print_data ? (
              <div className="spinner-border text-dark" role="status">
                <span className="sr-only">--</span>
              </div>
            ) : (
              <>
                {print_data && (
                  <div>
                    <p>The similarity between the 2 input sentences is...</p>
                    <h1>{outputData}</h1>
                    <p>with a conformal prediction interval of:</p>
                    <h1>
                      [{outputCPData[0].toFixed(4)} -{" "}
                      {outputCPData[1].toFixed(4)}]
                    </h1>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Body;
