import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Start from "./components/Start";
import About from "./components/About";
import React, { useEffect } from 'react';
import Login from "./components/Login";
import InsertData from "./components/InsertData";
import Register from "./components/Register";
import Footer from "./components/Footer";
import DetailsAccount from "./components/DetailsAccount";
import DownloadResult from "./components/DownloadResult";
import Profile from "./components/Profile";
import {RouterProvider, createBrowserRouter} from 'react-router-dom';
import Home from "./components/Home";
import ProfilePage from "./components/ProfilePage";
import Updates from "./components/Updates";
import Results from "./components/Results";

const router = createBrowserRouter([
  {
    path:"/", 
    element: <Home />,
  },
  {
    path:"/login",
    element: <Login />,
  },
  {
    path:"/register",
    element:<Register />,
  },
  {
    path:"/upload-file",
    element: <InsertData />,
  },
  {
    path:"/download-file",
    element:<DownloadResult />
  },
  {
    path:"/profile",
    element:<ProfilePage />
  },
  {
    path:"/basic-info",
    element:<DetailsAccount />,
  },
  {
    path:"/profile/updates",
    element: <Updates />
  },
  {
    path:"/profile/results",
    element: <Results />
  },



])

function App(){

  useEffect(() => {
    localStorage.removeItem('token'); // Assuming you store the token in localStorage
    localStorage.removeItem('user');  // Assuming you store user info in localStorage
  }, []);

  useEffect(() => {
    fetch('http://localhost:5000/')
      .then(response => response.json())
      .then(data => {
        console.log('Data from  API:' + data.message);
      })
      .catch(error => {
      });
  }, []);


  return (
    <>

    <RouterProvider router={router} />
    </>
  )
}

export default App;