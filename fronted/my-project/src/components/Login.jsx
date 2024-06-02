import { useState } from 'react';
import axios from 'axios';
import { CiLogin } from "react-icons/ci";
import Navbar from './Navbar';
import Footer from "./Footer";

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  
  const sendLogin = () => {
    setLoading(true);
    const data = { username, password };
    console.log('Sending login data:', data);  // Debug log
    axios.post('http://localhost:5000/login', data)
      .then((res) => {
        console.log('Response:', res.data);  // Debug log
        setMessage(res.data.message);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error:', err);  // Debug log
        if (err.response && err.response.status === 401) {
          setMessage(err.response.data.message);
        } else {
          setMessage("An error occurred. Please try again.");
        }
        setLoading(false);
      });
  };

  return (
    <div>
      <Navbar />
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-md shadow-md w-full max-w-md">
        <h1 className="text-2xl font-bold mb-6 flex items-center justify-center">
          Login <CiLogin className="ml-2" />
        </h1>
        <div className="mb-4 w-full">
          <label className="flex items-center gap-2 w-full">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 16 16"
              fill="currentColor"
              className="w-4 h-4 opacity-70"
            >
              <path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM12.735 14c.618 0 1.093-.561.872-1.139a6.002 6.002 0 0 0-11.215 0c-.22.578.254 1.139.872 1.139h9.47Z" />
            </svg>
            <input
              type="text"
              placeholder="Username"
              className="w-full border-2 border-gray-300 rounded-md p-2 focus:border-green-500"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
        </div>
        <div className="mb-6 w-full">
          <label className="flex items-center gap-2 w-full">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 16 16"
              fill="currentColor"
              className="w-4 h-4 opacity-70"
            >
              <path
                fillRule="evenodd"
                d="M14 6a4 4 0 0 1-4.899 3.899l-1.955 1.955a.5.5 0 0 1-.353.146H5v1.5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5v-2.293a.5.5 0 0 1 .146-.353l3.955-3.955A4 4 0 1 1 14 6Zm-4-2a.75.75 0 0 0 0 1.5.5.5 0 0 1 .5.5.75.75 0 0 0 1.5 0 2 2 0 0 0-2-2Z"
                clipRule="evenodd"
              />
            </svg>
            <input
              type="password"
              placeholder="Password"
              className="w-full border-2 border-gray-300 rounded-md p-2 focus:border-green-500"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
        </div>
        <button
          onClick={() => sendLogin()}
          className="bg-emerald-500 text-white py-2 px-4 rounded hover:bg-emerald-200 transition-colors duration-300 w-full"
        >
          Submit
        </button>
        {loading && (  
          <div className="mt-4 flex space-x-2">
            <span className="loading loading-spinner text-success"></span>
          </div>
        )}
        {message && <p className="mt-4 text-green-500">{message}</p>}
      </div>
    </div>
    <Footer />
    </div>
  );
};

export default Login;
