import React, { useEffect, useState } from 'react';
import axios from 'axios';
import ProfileNavbar from './ProfileNavbar';
import Footer from './Footer';
import { ImFileExcel } from "react-icons/im";

const Results = () => {
    const [results, setResults] = useState([]);

    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await axios.get('http://localhost:5000/profile/results');
                console.log('Server Response:', response.data);
                setResults(response.data.results);
            } catch (error) {
                console.log("Error:", error);
            }
        };
        fetchResults();
    }, []);

    const downloadFile = async (fileUrl) => {
        console.log('Downloading file from URL:', fileUrl);
        const response = await axios({
            url: fileUrl,
            method: 'GET',
            responseType: 'blob',
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'predictions.xlsx');
        document.body.appendChild(link);
        link.click();
        link.remove(); // Clean up the <a> element after download
    };

    return (
        <>
            <ProfileNavbar />
            <div className="mt-8 h-screen messages-container p-4 bg-gray-100 rounded-md shadow-lg">
                <h2 className="text-2xl font-semibold mb-4 flex items-center">
                    Previous Results
                    <span className="ml-2"><ImFileExcel className="h-7 w-7" /></span>
                </h2>
                {Array.isArray(results) && results.length > 0 ? (
                    <ul className="space-y-2">
                        {results.map((res, index) => (
                            <li key={index} className="bg-white p-4 mb-4 rounded-md shadow-md">
                                <p>{res.filename}</p>
                                {res.fileUrl ? (
                                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-2" onClick={() => downloadFile(res.fileUrl)}>
                                        Download File
                                    </button>
                                ) : (
                                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-2" onClick={() => downloadFile(`http://localhost:5000/download/${res.filename}`)}>
                                        Download File
                                    </button>
                                )}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-gray-500">No results yet</p>
                )}
            </div>
            <Footer />
        </>
    );
};

export default Results;
