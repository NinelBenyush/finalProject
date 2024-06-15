import React, { useState, useEffect } from "react";
import axios from "axios";
import ProfileNavbar from "./ProfileNavbar";
import Footer from "./Footer";

function Messages() {
    const [messages, setMessages] = useState([]);

    useEffect(() => {
        async function fetchData() {
            try {
                const [uploadedFilesRes, loginRes, downloadRes] = await Promise.all([
                    axios.get('http://localhost:5000/uploaded-files'),
                    axios.get('http://localhost:5000/get-login'),
                    axios.get('http://localhost:5000/getDownload')
                ]);

                const uploadedFiles = uploadedFilesRes.data.results.map(file => ({
                    ...file,
                    type: 'uploadedFile',
                    time: file.upload_time
                }));
                
                const greetings = loginRes.data.results.map(user => ({
                    ...user,
                    type: 'greeting',
                    time: user.login_time
                }));
                
                const downloads = downloadRes.data.results.map(dFile => ({
                    ...dFile,
                    type: 'downloadedFile',
                    time: dFile.download_time
                }));

                const combinedMessages = [...uploadedFiles, ...greetings, ...downloads];
                combinedMessages.sort((a, b) => new Date(a.time) - new Date(b.time));

                setMessages(combinedMessages);
            } catch (error) {
                console.error("Error fetching data:", error);
            }
        }

        fetchData();
    }, []);

    const renderMessage = (message, index) => {
        switch (message.type) {
            case 'uploadedFile':
                return (
                    <li role="article" key={index} className="relative pl-6">
                        <span className="absolute left-0 z-10 flex items-center justify-center w-8 h-8 -translate-x-1/2 rounded-full bg-slate-200 text-slate-700 ring-2 ring-white ">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth="1.5"
                                stroke="currentColor"
                                className="w-4 h-4"
                                role="presentation"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M6 6.878V6a2.25 2.25 0 012.25-2.25h7.5A2.25 2.25 0 0118 6v.878m-12 0c.235-.083.487-.128.75-.128h10.5c.263 0 .515.045.75.128m-12 0A2.25 2.25 0 004.5 9v.878m13.5-3A2.25 2.25 0 0119.5 9v.878m0 0a2.246 2.246 0 00-.75-.128H5.25c-.263 0-.515.045-.75.128m15 0A2.25 2.25 0 0121 12v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6c0-.98.626-1.813 1.5-2.122"
                                />
                            </svg>
                        </span>
                        <div className="flex flex-col flex-1 gap-0">
                            <h4 className="text-sm font-medium text-slate-700">
                               File {message.fileName} uploaded successfully
                            </h4>
                            <p className="text-xs text-slate-500">{message.upload_time}</p>
                        </div>
                    </li>
                );
            case 'greeting':
                return (
                    <li role="article" key={index} className="relative pl-6">
                        <span className="absolute left-0 z-10 flex items-center justify-center w-8 h-8 -translate-x-1/2 rounded-full bg-slate-200 text-slate-700 ring-2 ring-white ">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth="1.5"
                                stroke="currentColor"
                                className="w-4 h-4"
                                role="presentation"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M6 6.878V6a2.25 2.25 0 012.25-2.25h7.5A2.25 2.25 0 0118 6v.878m-12 0c.235-.083.487-.128.75-.128h10.5c.263 0 .515.045.75.128m-12 0A2.25 2.25 0 004.5 9v.878m13.5-3A2.25 2.25 0 0119.5 9v.878m0 0a2.246 2.246 0 00-.75-.128H5.25c-.263 0-.515.045-.75.128m15 0A2.25 2.25 0 0121 12v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6c0-.98.626-1.813 1.5-2.122"
                                />
                            </svg>
                        </span>
                        <div className="flex flex-col flex-1 gap-0">
                            <h4 className="text-sm font-medium text-slate-700">
                               Welcome back {message.username}
                            </h4>
                            <p className="text-xs text-slate-500">{message.login_time}</p>
                        </div>
                    </li>
                );
            case 'downloadedFile':
                return (
                    <li role="article" key={index} className="relative pl-6">
                        <span className="absolute left-0 z-10 flex items-center justify-center w-8 h-8 -translate-x-1/2 rounded-full bg-slate-200 text-slate-700 ring-2 ring-white ">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth="1.5"
                                stroke="currentColor"
                                className="w-4 h-4"
                                role="presentation"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"
                                />
                            </svg>
                        </span>
                        <div className="flex flex-col flex-1 gap-0">
                            <h4 className="text-sm font-medium text-slate-700">
                                The result file {message.filename} downloaded successfully
                            </h4>
                            <p className="text-xs text-slate-500">{message.download_time}</p>
                        </div>
                    </li>
                );
            default:
                return null;
        }
    };

    return (
        <>
        <ProfileNavbar/>
        <div className="flex justify-center items-center  bg-gray-100">
        <div className="bg-slate-100 p-8 rounded-lg s w-full max-w-4xl">
        <h3 className="text-left text-2xl font-bold mb-4">Messages</h3>
        <ul
            aria-label="Activity feed"
            role="feed"
            className="relative flex  flex-col gap-12 py-12 pl-6 before:absolute before:top-0 before:left-6 before:h-full before:-translate-x-1/2 before:border before:border-dashed before:border-slate-200 after:absolute after:top-6 after:left-6 after:bottom-6 after:-translate-x-1/2 after:border after:border-slate-200 "
        >
            {messages.map((message, index) => renderMessage(message, index))}
        </ul>
        </div>
        </div>
        <Footer />
        </>
    );
}

export default Messages;
