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
import UploadNewFile from "./components/UploadNewFile";
import UploadedFiles from "./components/UploadedFiles";
import Messages from "./components/Messages";
import Graph from "./components/Graph";
import ErrorPage from "./components/ErrorPage";

const router = createBrowserRouter([
  {
    path:"/", 
    element: <Home />,
    errorElement:<ErrorPage />
  },
  {
    path:"/login",
    element: <Login />,
    errorElement:<ErrorPage />
  },
  {
    path:"/register",
    element:<Register />,
    errorElement:<ErrorPage />
  },
  {
    path:"/upload-file",
    element: <InsertData />,
    errorElement:<ErrorPage />
  },
  {
    path:"/download-file",
    element:<DownloadResult />,
    errorElement:<ErrorPage />
  },
  {
    path:"/basic-info",
    element:<DetailsAccount />,
    errorElement:<ErrorPage />
  },
  {
    path:"/profile",
    element:<ProfilePage />,
    errorElement:<ErrorPage />
  },

  {
    path:"/profile/updates",
    element: <Updates />,
    errorElement:<ErrorPage />
  },
  {
    path:"/profile/results",
    element: <Results />,
    errorElement:<ErrorPage />
  },
  {
    path:"/UploadNewFile",
    element: <UploadNewFile />,
    errorElement:<ErrorPage />
  },
  {
    path:"/profile/Files",
    element: <UploadedFiles />,
    errorElement:<ErrorPage />
  },
  {
    path:"/check",
    element :<Graph />,
    errorElement:<ErrorPage />
  },
  {
    path:"/profile/messages",
    element: <Messages />,
    errorElement :<ErrorPage />
  }
  

])

function App(){

  useEffect(() => {
    localStorage.removeItem('token'); 
    localStorage.removeItem('user');  
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
