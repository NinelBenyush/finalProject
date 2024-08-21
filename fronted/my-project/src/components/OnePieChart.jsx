import React, { useEffect, useRef } from 'react';
import { Chart, PieController, ArcElement, Tooltip, Legend, Title } from 'chart.js';

Chart.register(PieController, ArcElement, Tooltip, Legend, Title);
//section on the home page
const OnePieChart = () => {
  const chartRef = useRef(null);

  useEffect(() => {
    const ctx = chartRef.current.getContext('2d');

    let chartInstance = null;

    if (chartInstance) {
      chartInstance.destroy();
    }
    
    const dataPie = {
      labels: ["January-March", "April-June", "July-September", "October-December"],
      datasets: [
        {
          label: "Sequences of 3 months",
          data: [25, 25, 25, 25],
          backgroundColor: [
            "rgb(201, 240, 77)",
            "rgb(143, 217, 76)",
            "rgb(81, 209, 71)",
            "rgb(175, 242, 138)",
          ],
          hoverOffset: 4,
        },
      ],
    };

    const configPie = {
      type: 'pie',
      data: dataPie,
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Given months',
            font: {
              size: 20
            },
            padding: {
              top: 10,
            
            }
          },
          tooltip: {
            callbacks: {
              label: function (tooltipItem) {
                return tooltipItem.label; 
              }
            }
          }
        }
      }
    };

    new Chart(ctx, configPie);
  }, []);

  return (
    <div className=" rounded-lg overflow-hidden">
      <canvas style={{ width: '10px', height: '10px' }}  ref={chartRef} width={200} height={200}></canvas>
    </div>
  );
};

export default OnePieChart;
