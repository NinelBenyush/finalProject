import React, { useState } from 'react';
import axios from 'axios';
import { FaFileDownload } from "react-icons/fa";

const FileDownload = () => {
    const [message, setMessage] = useState('');

    const downloadFile = () => {
        axios.get('http://localhost:5000', {
            responseType: 'blob'  // Important for handling binary data
        })
        .then((response) => {
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const a = document.createElement('a');
            a.href = url;
            a.download = 'DataPrediction.csv'; // You can set the filename here
            document.body.appendChild(a);
            a.click();
            a.remove();
            setMessage('File downloaded successfully.');

        })
        .catch((error) => {
            console.error('Error:', error);
            setMessage("An error occurred. Please try again.");

        });
    };

    return (
        <div>
       <div className="hero min-h-screen bg-base-200">
        <div className="hero-content text-center">
        <div className="max-w-md">
         <h1 className="text-5xl font-bold">Here are your predictions <FaFileDownload className="inline-block text-4xl" /></h1>
         <p className="py-6"> We have processed your data and generated the prediction results <br></br>Please go ahead and click on the download button</p>
         <button onClick={downloadFile} className="btn bg-emerald-400">Download</button>
         {message && <p>{message}</p>}
        </div>
       </div>
     </div>
     </div>

    );
};

export default FileDownload;
