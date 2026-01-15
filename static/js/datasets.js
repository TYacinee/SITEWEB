// Loading data from flask
fetch("/data")
  .then((response) => response.json())
  .then((rawData) => {
    const data = rawData.map((d) => ({
      ...d,
      result:
        d.result === "winner"
          ? "win"
          : d.result === "loser"
          ? "loss"
          : d.result,
    }));
    const wins = data.filter((d) => d.result === "win");
    const losses = data.filter((d) => d.result === "loss");

    // Creation of graph 1 – SHOOTING EFFICIENCY
    function scatterTrace(dataset, color, name) {
      return {
        x: dataset.map((d) => Number(d.shots) || 0),
        y: dataset.map((d) => Number(d.goals) || 0),
        mode: "markers",
        type: "scatter",
        name: name,
        marker: {
          size: 10,
          color: color,
          opacity: 0.75,
        },
        text: dataset.map(
          (d) =>
            `Shots: ${d.shots}<br>
             Goals: ${d.goals}<br>
             Shooting %: ${d["shooting percentage"]}`
        ),
        hoverinfo: "text",
      };
    }

    Plotly.newPlot(
      "plot1",
      [
        scatterTrace(data, "#00e5ff", "All matches"),
        scatterTrace(wins, "#00ff9c", "Wins"),
        scatterTrace(losses, "#ff5252", "Losses"),
      ],
      {
        title: "Shooting Efficiency – Shots vs Goals",
        xaxis: { title: "Shots" },
        yaxis: { title: "Goals" },
        updatemenus: [
          {
            buttons: [
              {
                method: "update",
                args: [{ visible: [true, false, false] }],
                label: "All",
              },
              {
                method: "update",
                args: [{ visible: [false, true, false] }],
                label: "Wins",
              },
              {
                method: "update",
                args: [{ visible: [false, false, true] }],
                label: "Losses",
              },
            ],
            direction: "right",
            x: -0.05,
            xanchor: "right",
            y: 1,
            yanchor: "top",
            bgcolor: "#0a1a22",
            bordercolor: "#00e5ff",
            borderwidth: 1,
            font: { color: "inherit" },
          },
        ],
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "white" },
        autosize: true,
      },
      { responsive: true }
    );

    // Creation of graph 2 – BOOST MANAGEMENT
    const total = (key) => data.reduce((s, d) => s + (Number(d[key]) || 0), 0);
    const avg = (key) => total(key) / data.length;

    Plotly.newPlot(
      "plot2",
      [
        {
          x: ["Collected", "Used", "Stolen"],
          y: [
            total("amount collected"),
            total("amount used while supersonic"),
            total("amount stolen"),
          ],
          type: "bar",
          marker: { color: ["#00e5ff", "#00bcd4", "#00838f"] },
        },
      ],
      {
        title: "Boost Management",
        updatemenus: [
          {
            buttons: [
              {
                label: "Total",
                method: "restyle",
                args: [
                  "y",
                  [
                    [
                      total("amount collected"),
                      total("amount used while supersonic"),
                      total("amount stolen"),
                    ],
                  ],
                ],
              },
              {
                label: "Average per match",
                method: "restyle",
                args: [
                  "y",
                  [
                    [
                      avg("amount collected"),
                      avg("amount used while supersonic"),
                      avg("amount stolen"),
                    ],
                  ],
                ],
              },
            ],
            direction: "right",
            x: -0.05,
            xanchor: "right",
            y: 1,
            yanchor: "top",
            bgcolor: "#0a1a22",
            bordercolor: "#00e5ff",
            borderwidth: 1,
            font: { color: "inherit" },
          },
        ],
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "white" },
        autosize: true,
      },
      { responsive: true }
    );

    // Creation of graph 3 – WHAT DRIVES WINNING
    function avgMetric(dataset, key) {
      return (
        dataset.reduce((s, d) => s + (Number(d[key]) || 0), 0) / dataset.length
      );
    }

    Plotly.newPlot(
      "plot3",
      [
        {
          x: ["Goals", "Shots", "Saves", "Demos"],
          y: [
            avgMetric(wins, "goals"),
            avgMetric(wins, "shots"),
            avgMetric(wins, "saves"),
            avgMetric(wins, "demos inflicted"),
          ],
          name: "Wins",
          type: "bar",
          marker: { color: "#00ff9c" },
        },
        {
          x: ["Goals", "Shots", "Saves", "Demos"],
          y: [
            avgMetric(losses, "goals"),
            avgMetric(losses, "shots"),
            avgMetric(losses, "saves"),
            avgMetric(losses, "demos inflicted"),
          ],
          name: "Losses",
          type: "bar",
          marker: { color: "#ff5252" },
        },
      ],
      {
        title: "What Changes Between Wins and Losses?",
        barmode: "group",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "white" },
        autosize: true,
      },
      { responsive: true }
    );
  });
