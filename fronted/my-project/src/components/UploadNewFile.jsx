import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function UploadNewFile() {
    const navigate = useNavigate();
    const [filename, setFilename] = useState('');
    const [description, setDescription] = useState('');
    const [response, setResponse] = useState(null);

    const handleUpload = async (e) => {
        e.preventDefault(); 
        try {
            const res = await axios.post('http://localhost:5000/profile', {
                    filename,
                    description
            });
            setResponse(res.data);
            navigate('/upload-file');
            if (res.data.status === 'success') {
                navigate('/upload-file');
            }
        } catch (error) {
            console.error('There was an error', error);
        }
    };

    return (
        <div className="m-10">
            <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
            <form onSubmit={handleUpload}>
                <div className="space-y-12">
                    <div className="border-b border-gray-900/10 pb-12">
                        <h2 className="text-base font-semibold leading-7 text-gray-900">Upload File</h2>
                        <p className="mt-1 text-sm leading-6 text-gray-600">
                            Here update the name of the file, the date and the description of the data
                        </p>

                        <div className="mt-10 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-6">
                            <div className="sm:col-span-4">
                                <label htmlFor="filename" className="block text-sm font-medium leading-6 text-gray-900">
                                    File name
                                </label>
                                <div className="mt-2">
                                    <div className="flex rounded-md shadow-sm ring-1 ring-inset ring-gray-300 focus-within:ring-2 focus-within:ring-inset focus-within:ring-green-600 sm:max-w-md">
                                        <input
                                            type="text"
                                            name="filename"
                                            value={filename}
                                            autoComplete="filename"
                                            className="block flex-1 border-0 bg-transparent py-1.5 pl-1 text-gray-900 placeholder:text-gray-400 focus:ring-0 sm:text-sm sm:leading-6"
                                            placeholder=".xlsx"
                                            onChange={(e) => setFilename(e.target.value)}
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="col-span-full">
                                <label htmlFor="description" className="block text-sm font-medium leading-6 text-gray-900">
                                    Description
                                </label>
                                <div className="mt-2">
                                    <textarea
                                        name="description"
                                        value={description}
                                        rows={3}
                                        className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-green-600 sm:text-sm sm:leading-6"
                                        onChange={(e) => setDescription(e.target.value)}
                                    />
                                </div>
                                <p className="mt-3 text-sm leading-6 text-gray-600">Write a few sentences about the data that's inside the file.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="mt-6 flex items-center justify-end gap-x-6">
                    <button type="button" className="text-sm font-semibold leading-6 text-gray-900">
                        Cancel
                    </button>
                    <button
                        type="submit"
                        className="rounded-md bg-emerald-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
                    >
                        Save
                    </button>
                </div>
            </form>
            </div>
        </div>
    );
}

export default UploadNewFile;
