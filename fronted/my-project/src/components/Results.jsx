import React, { useEffect, useState } from 'react';
import axios from 'axios';
import ProfileNavbar from './ProfileNavbar';
import Footer from './Footer';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/20/solid';
import { ImFileExcel } from "react-icons/im";

const Results = () => {
    const [results, setResults] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const resultsPerPage = 6;

    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await axios.get('http://localhost:5000/profile/results');
                console.log('Server Response:', response.data);
                const sortedResults = response.data.results.sort((a, b) => new Date(b.date) - new Date(a.date));
                setResults(sortedResults);
            } catch (error) {
                console.log("Error:", error);
            }
        };
        fetchResults();
    }, []);

    const downloadFile = async (filename) => {
        console.log('Downloading file:', filename);
        try {
            const response = await axios.get(`http://localhost:5000/profile/results/${filename}.csv`, { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${filename}.csv`); // Make sure to append the extension here
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error("Error downloading file:", error);
        }
    };

    const indexOfLastResult = currentPage * resultsPerPage;
    const indexOfFirstResult = indexOfLastResult - resultsPerPage;
    const currentResults = results.slice(indexOfFirstResult, indexOfLastResult);
    const totalPages = Math.ceil(results.length / resultsPerPage);

    return (
        <>
            <ProfileNavbar />
            <div className="overflow-x-auto h-screen m-10">
                <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
                    <h2 className="text-2xl font-semibold mb-4 flex items-center">
                        Previous Results
                        <span className="ml-2"><ImFileExcel className="h-7 w-7" /></span>
                    </h2>
                    <div className="w-full">
                        <table className="table w-full">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Filename</th>
                                    <th>Date</th>
                                    <th>Download</th>
                                </tr>
                            </thead>
                            <tbody>
                                {currentResults.map((res, index) => (
                                    <tr key={index}>
                                        <th>{indexOfFirstResult + index + 1}</th>
                                        <td>{res.filename}</td>
                                        <td>{new Date(res.date).toLocaleDateString()}</td>
                                        <td>
                                            <button className="bg-emerald-500 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded" onClick={() => downloadFile(res.filename)}>
                                                Download File
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="flex items-center justify-between border-t border-gray-200 bg-white px-4 py-3 sm:px-6">
                        <div className="flex flex-1 justify-between sm:hidden">
                            <button
                                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                                disabled={currentPage === 1}
                                className="relative inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                            >
                                Previous
                            </button>
                            <button
                                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                                disabled={currentPage === totalPages}
                                className="relative ml-3 inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                            >
                                Next
                            </button>
                        </div>
                        <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between w-full">
                            <div>
                                <p className="text-sm text-gray-700">
                                    Showing <span className="font-medium">{indexOfFirstResult + 1}</span> to <span className="font-medium">{Math.min(indexOfLastResult, results.length)}</span> of{' '}
                                    <span className="font-medium">{results.length}</span> results
                                </p>
                            </div>
                            <div>
                                <nav className="isolate inline-flex -space-x-px rounded-md shadow-sm" aria-label="Pagination">
                                    <button
                                        onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                                        disabled={currentPage === 1}
                                        className="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0"
                                    >
                                        <span className="sr-only">Previous</span>
                                        <ChevronLeftIcon className="h-5 w-5" aria-hidden="true" />
                                    </button>
                                    {[...Array(totalPages)].map((_, index) => (
                                        <button
                                            key={index}
                                            onClick={() => setCurrentPage(index + 1)}
                                            className={`relative inline-flex items-center px-4 py-2 text-sm font-semibold ${currentPage === index + 1 ? 'bg-emerald-600 text-white' : 'text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50'} focus:z-20 focus:outline-offset-0`}
                                        >
                                            {index + 1}
                                        </button>
                                    ))}
                                    <button
                                        onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                                        disabled={currentPage === totalPages}
                                        className="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0"
                                    >
                                        <span className="sr-only">Next</span>
                                        <ChevronRightIcon className="h-5 w-5" aria-hidden="true" />
                                    </button>
                                </nav>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Results;
