import React, { useState } from 'react';
import { TbFileSearch } from "react-icons/tb";
import { IoMdNotifications } from "react-icons/io";
import { IoInformationCircleSharp } from "react-icons/io5";

const Sidebar = () => {
  const [sidenav, setSidenav] = useState(true);

  const handleSidebarToggle = () => {
    setSidenav(!sidenav);
  };

  return (
    <div className="font-poppins antialiased">
      <div id="view" className="h-full w-screen flex flex-row">
        <button
          onClick={handleSidebarToggle}
          className="p-2 border-2 bg-white rounded-md border-gray-200 shadow-lg text-gray-500 focus:bg-teal-500 focus:outline-none focus:text-white absolute top-0 left-0 sm:hidden"
        >
          <svg
            className="w-5 h-5 fill-current"
            fill="currentColor"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              fillRule="evenodd"
              d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
              clipRule="evenodd"
            ></path>
          </svg>
        </button>
        {sidenav && (
          <div
            id="sidebar"
            className=" bg-emerald-100 h-screen md:block shadow-xl px-3 w-30 md:w-60 lg:w-60 overflow-x-hidden transition-transform duration-300 ease-in-out"
          >
            <div className="space-y-6 md:space-y-10 mt-10">
              <div id="profile" className="space-y-3 flex flex-col items-center">
                <div className="avatar placeholder">
                  <div className="bg-neutral w-8 md:w-16 rounded-full text-neutral-content flex items-center justify-center">
                    <span>A</span>
                  </div>
                </div>
                <div>
                  <h2 className="font-medium text-xs md:text-sm text-center text-black">
                    username
                  </h2>
                </div>
              </div>

              <div id="menu" className="flex flex-col space-y-2">
                <a
                  href="#"
                  className="text-sm font-medium text-gray-700 py-2 px-2 hover:bg-emerald-400 hover:text-white hover:text-base rounded-md transition duration-150 ease-in-out"
                >
                  <TbFileSearch className="w-6 h-6 inline-block" />
                  <span> Results</span>
                </a>
                <a
                  href="#"
                  className="text-sm font-medium text-gray-700 py-2 px-2 hover:bg-emerald-400 hover:text-white hover:scale-105 rounded-md transition duration-150 ease-in-out"
                >
                  <IoMdNotifications className="w-6 h-6 inline-block" />
                  <span>Updates</span>
                </a>
                <a
                  href="#"
                  className="text-sm font-medium text-gray-700 py-2 px-2 hover:bg-emerald-400 hover:text-white hover:text-base rounded-md transition duration-150 ease-in-out"
                >
                  <IoInformationCircleSharp className="w-6 h-6 inline-block" />
                  <span> Details</span>
                </a>
              </div>
            </div>
          </div>
        )}
        <div className="flex flex-col w-full">
          <div className="overflow-auto h-screen pb-24 px-4 md:px-6">
            <h1 className="text-4xl font-semibold text-gray-800 dark:text-white">
              Good {new Date().getHours() < 12 ? 'Morning' : new Date().getHours() < 18 ? 'Afternoon' : 'Evening'}.
            </h1>
            <h2 className="text-md text-gray-400">Welcome back</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 p-4 gap-4">
              {/* Insert your content here */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
