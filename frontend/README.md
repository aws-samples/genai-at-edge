# GenAI at Edge

## Visualization
The output of the system can be visualized as a locally hosted webpage. This is done using either a HTML webpage or StreamLit solution.
Note: The assumes the AWS Lambda, AWS IoT Rule, Amazon API GateWay are all setup correctly as shown in the [AWS Architecture](../assets/AWSArchitecture.png).

### 1. HTML Webpage:
- Host the webpage `index-nanovlm.html` or `index-fastersam.html` locally.

### 2. StreamLit App:
- Run a streamlit app as follows:
    ```
    $ pip3 install streamlit
    $ [NanoVLM] streamlit run nanovlm-app.py
    $ [FasterSAM] streamlit run fastersam-app.py
    ```