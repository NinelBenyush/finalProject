import { PaperClipIcon } from '@heroicons/react/20/solid'
import React, {useEffect, useState} from 'react';
import axios from 'axios';
import { IoPerson } from "react-icons/io5";

function PersonalArea(){
  const [ info, setInfo] =useState([]);

  useEffect(() => {
    async function fetchInfo(){
      try{
        const respone = await axios.get('http://localhost:5000/profile');
        if(respone.data.status === 'success'){
          setInfo(respone.data.info[0]);
        }else{
          console.error('failed to fetch info', respone.data.message);
        }
      }catch(error){
        console.error('error', error);
      }
    }
    fetchInfo();
  }, []);

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
            <dd className="mt-1 text-sm leading-6 text-gray-700 sm:col-span-2 sm:mt-0">example@example.com</dd>
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