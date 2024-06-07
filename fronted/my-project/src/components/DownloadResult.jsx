import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FaFileDownload } from "react-icons/fa";
import MiniNavbar from "./MiniNabvar";
import Footer from './Footer';

const FileDownload = () => {
    const [message, setMessage] = useState('');
    const [successDownload, setSuccessDownload] = useState(false);

    const downloadFile = () => {
        axios.get('http://localhost:5000/download-file', {
            responseType: 'blob'  // Important for handling binary data
        })
        .then((response) => {
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const a = document.createElement('a');
            a.href = url;
            a.download = 'DataPrediction.csv'; 
            document.body.appendChild(a);
            a.click();
            a.remove();
            setSuccessDownload(true);
        })
        .catch((error) => {
            console.error('Error:', error);
            setMessage("An error occurred. Please try again.");
        });
    };

    useEffect(() => {
        if (successDownload) {
            const timer = setTimeout(() => {
                setSuccessDownload(false);
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [successDownload]);

    return (
        <div>
            <MiniNavbar />
            <div className="hero min-h-screen bg-base-200">
                <div className="hero-content text-center">
                    <div className="max-w-md">
                        {successDownload && (
                            <div className="flex justify-center mb-4">
                                <div id="toast-success" className="flex items-center w-full max-w-xs p-4 text-gray-500 bg-white rounded-lg shadow dark:text-gray-400 dark:bg-gray-800" role="alert">
                                    <div className="inline-flex items-center justify-center flex-shrink-0 w-8 h-8 text-green-500 bg-green-100 rounded-lg dark:bg-green-800 dark:text-green-200">
                                        <svg className="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5Zm3.707 8.207-4 4a1 1 0 0 1-1.414 0l-2-2a1 1 0 0 1 1.414-1.414L9 10.586l3.293-3.293a1 1 0 0 1 1.414 1.414Z" />
                                        </svg>
                                        <span className="sr-only">Check icon</span>
                                    </div>
                                    <div className="ml-3 text-sm font-normal">File downloaded successfully</div>
                                </div>
                            </div>
                        )}
                        <h1 className="text-5xl font-bold">Here are your predictions <FaFileDownload className="inline-block text-4xl" /></h1>
                        <p className="py-6"> We have processed your data and generated the prediction results <br />Please go ahead and click on the download button</p>
                        <button onClick={downloadFile} className="btn bg-emerald-400">Download</button>
                        {message && <p>{message}</p>}
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default FileDownload;
