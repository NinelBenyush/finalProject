import React from 'react';
import { IoFootstepsOutline } from "react-icons/io5";
import step1Img from "../assets/step1Img.png";
import column from "../assets/column.png";

function TimeLine() {
  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md relative">
      <div className="px-4 sm:px-0">
        <h3 className="text-base font-semibold leading-7 text-gray-900">
          Steps to upload the file <IoFootstepsOutline className="inline-block text-xl ml-1"/>
        </h3>
      </div>
      <ul className="timeline timeline-snap-icon max-md:timeline-compact timeline-vertical">
      <li className="relative">
  <div className="timeline-middle">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
    </svg>
  </div>
  <div className="timeline-start md:text-end mb-10 relative">
    <time className="font-mono italic">Step 1</time>
    <div className="text-lg font-black">File extension</div>
    Please ensure that you upload the data file in Excel format with the .xlsx extension.
  </div>
  <img src={step1Img} alt="Step 1" className="h-40 w-40 absolute right-14 top-1/2 -translate-y-1/2" />
  <hr />
</li>

        <li>
          <hr />
          <div className="timeline-middle">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="timeline-end mb-10">
            <time className="font-mono italic">Step 2</time>
            <div className="text-lg font-black">File name</div>
            Make sure to include the year in the file name, for example, sales_2023.xlsx.
          </div>
          <hr />
        </li>
        <li>
          <hr />
          <div className="timeline-middle">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="timeline-start md:text-end mb-10">
            <time className="font-mono italic">Step 3</time>
            <div className="text-lg font-black">Column names</div>
            Please ensure that the column is renamed as follows:
            "catalog number" should be renamed as "code"
            "color of the product" should be renamed as "color"
            The names of the months should start with capital letters.
            The amount of sales for each month should be labeled as "Value".
            The sum of the inventory should be labeled as "Inventory".
          </div>
          <img src={column} alt="Step 1" className="h-40 w-40 absolute right-14 top-1/2 -translate-y-1/2" />
          <hr />
        </li>
      </ul>
    </div>
  );
}

export default TimeLine;
