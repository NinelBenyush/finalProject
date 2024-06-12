import { useState } from 'react';
import axios from 'axios';
import MiniNavbar from "./MiniNabvar";
import Footer from './Footer';
import { useNavigate } from 'react-router-dom';
import registerImg from "../assets/registerImg.svg";

function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] =useState('');
  const [ email, setEmail] = useState('');
  const  [loading, setLoading] = useState(false);
  const [ message, setMessage] = useState('');
  const [emailError, setEmailError] = useState('');
  const navigate = useNavigate(); 

  const validateEmail = (email) => {
    if (!email.includes('@')) {
      setEmailError('Email is not valid.');
    } else {
      setEmailError('');
    }
  };
  
  const sendRegister = () => {
    if(password!== confirmPassword){
      setMessage("passwords dont match")
      return;
    }
    if (!email.includes('@')) {
      setMessage('Email is not valid.');
      return;
    }
    setLoading(true)
    const data = {username, password, email};
    console.log('Sending register data:', data); 

    axios.post('http://localhost:5000/register', data)
    .then((res) => {
      console.log('reponse', res.data);
      setMessage(res.data.message);
      setLoading(false);
      navigate("/profile");
    })
    .cacth((err) => {
      console.log("error",err);
      if(err.response && err.response.status===400){
        setMessage(err.response.data.message);
      }
      else {
        setMessage("An error occurred. Please try again.");
      }
      setLoading(false);
    })

  }


    return (
      <div>
        <MiniNavbar/>
      <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="hidden lg:flex lg:w-1/2">
          <img
            src={registerImg}
            alt="Placeholder Image"
            className="object-cover mx-auto w-1/2 lg:w-2/3 md:w-3/4 h-auto"
          />
        </div>
        <div className="p-8 rounded-md w-full max-w-md lg:max-w-none lg:w-1/2 lg:bg-transparent bg-white">
          <h1 className="text-2xl font-bold mb-6 flex items-center justify-center">
            Register
          </h1>
          <div className="mb-4 w-full">
            <label className="flex items-center gap-2 w-full">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 16 16"
                fill="currentColor"
                className="w-4 h-4 opacity-70"
              >
                <path d="M2.5 3A1.5 1.5 0 0 0 1 4.5v.793c.026.009.051.02.076.032L7.674 8.51c.206.1.446.1.652 0l6.598-3.185A.755.755 0 0 1 15 5.293V4.5A1.5 1.5 0 0 0 13.5 3h-11Z" />
                <path d="M15 6.954 8.978 9.86a2.25 2.25 0 0 1-1.956 0L1 6.954V11.5A1.5 1.5 0 0 0 2.5 13h11a1.5 1.5 0 0 0 1.5-1.5V6.954Z" />
              </svg>
              <input
                type="text"
                placeholder="Email"
                className="w-full border-2 border-gray-300 rounded-md p-2 focus:border-green-500"
                value = {email}
                onChange={(e) => {
                  setEmail(e.target.value)
                    validateEmail(e.target.value);
                }}
              />
            </label>
            {emailError && <p className="mt-2 text-red-500">{emailError}</p>}
          </div>
  
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
                value = {username}
                onChange = {(e) => setUsername(e.target.value)}
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
                value = {password}
                onChange= {(e) => setPassword(e.target.value)}
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
                placeholder="enter password again"
                className="w-full border-2 border-gray-300 rounded-md p-2 focus:border-green-500"
                value = {confirmPassword}
                onChange = {(e) => setConfirmPassword(e.target.value)}
              />

            </label>
          </div>

          <button
          onClick = {() => sendRegister()}
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
      <Footer/>
      </div>
    );
  }
  
  export default Register;
  