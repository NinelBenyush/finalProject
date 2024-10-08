import { PaperClipIcon } from '@heroicons/react/20/solid'
import React, {useEffect, useState} from 'react';
import axios from 'axios';
import { IoPerson } from "react-icons/io5";

//section with the personal info 
function PersonalArea(){
  const [ info, setInfo] =useState([]);
  const username = localStorage.getItem('username');

  const [error, setError] = useState('');

  useEffect(() => {
      async function fetchInfo() {
          try {
              const response = await axios.get(`http://localhost:5000/profile?username=${username}`);
              
              if (response.data.status === 'success') {
                  setInfo(response.data.info);
              } else {
                  console.error('Failed to fetch info', response.data.message);
                  setError(response.data.message);
              }
          } catch (error) {
              console.error('Error fetching profile info:', error.response ? error.response.data : error.message);
              setError(error.response ? error.response.data.message : error.message);
          }
      }
      fetchInfo();
  }, [username]);

  if (error) {
      return <div>{error}</div>;
  }

    return (
        <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <div className="px-4 sm:px-0">
      <h3 className="text-base font-semibold leading-7 text-gray-900">Personal Details <IoPerson className="inline-block text-xl ml-1"/></h3>
      </div>
      <div className="mt-6 border-t border-gray-100">
        <dl className="divide-y divide-gray-100">
          <div className="px-4 py-6 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-0">
            <dt className="text-sm font-medium leading-6 text-gray-900">Full name</dt>
            <dd className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">{info.firstName} {info.lastName}</dd>
          </div>
          <div className="px-4 py-6 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-0">
            <dt className="text-sm font-medium leading-6 text-gray-900">Company Name</dt>
            <dd className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">{info.companyName}</dd>
          </div>
          <div className="px-4 py-6 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-0">
            <dt className="text-sm font-medium leading-6 text-gray-900">Email address</dt>
            <dd className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">{info.emailAddress}</dd>
          </div>
          <div className="px-4 py-6 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-0">
            <dt className="text-sm font-medium leading-6 text-gray-900">Phone Number</dt>
            <dd className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">{info.phoneNumber}</dd>
          </div>
          <div className="px-4 py-6 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-0">
            <dt className="text-sm font-medium leading-6 text-gray-900">Company Description</dt>
            <dd className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">
             {info.companyDescription}
            </dd>
          </div>

        </dl>
      </div>
    </div>
    )
}
export default PersonalArea;