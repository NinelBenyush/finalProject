import React,{useEffect}  from 'react';
import { TbFileInfo } from "react-icons/tb";
import resultsE from "../assets/resultsE.png";

function ResultsExplanation() {
    useEffect(() => {
        const textElement = document.querySelector('.text-with-line-breaks');
        if (textElement) {
          const text = textElement.innerHTML;
          const newText = text.replace(/\./g, '.<br />');
          textElement.innerHTML = newText;
        }
      }, []);


  return (
    <div className='m-10'>
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">

        <div className="px-4 sm:px-0">
          <h3 className="text-base font-semibold leading-7 text-gray-900">About the results <TbFileInfo className="inline-block text-xl ml-1" /> </h3>
        </div>

        <div className="mt-6 border-t border-gray-100">
          <dl className="divide-y divide-gray-100">

            <div className="px-4 py-6 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-0">
              <p className="text-sm leading-6 w-full text-gray-900 text-with-line-breaks sm:col-span-3">
                You will receive the results in an Excel file. The columns will display your products, with each row representing the upcoming months. The values inside the rows will indicate the inventory that you will need to order for each product per month.
              </p>
            </div>


            <div className="flex flex-col">
                  <div className="overflow-x-auto pl-4 sm:-mx-6 lg:-mx-8 ">
                    <div className="inline-block min-w-full py-2 sm:px-6 lg:px-8">
                      <div className="overflow-hidden">
                        <table className="min-w-full text-center text-sm font-light text-surface dark:text-white lg:block">
                          <thead className="border-b border-neutral-200 bg-neutral-50 font-medium dark:border-white/10 dark:text-neutral-800">
                            <tr>
                              <th scope="col" className="  bg-green-100 px-6 py-4 "></th>
                              <th scope="col" className="px-6 py-4 border-b border-success-200 bg-green-100 text-neutral-800">Product1</th>
                              <th scope="col" className="px-6 py-4 border-b border-success-200 bg-green-100 text-neutral-800">Product2</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-neutral-200 dark:border-white/10">
                              <td className="whitespace-nowrap px-6 py-4 font-medium">June</td>
                              <td className="whitespace-nowrap px-6 py-4">12</td>
                              <td className="whitespace-nowrap px-6 py-4">24</td>
                            </tr>
                            <tr className="border-b border-neutral-200 dark:border-white/10">
                              <td className="whitespace-nowrap px-6 py-4 font-medium">July</td>
                              <td className="whitespace-nowrap px-6 py-4">42</td>
                              <td className="whitespace-nowrap px-6 py-4">23</td>
                            </tr>
                            <tr className="border-b border-neutral-200 dark:border-white/10">
                              <td className="whitespace-nowrap px-6 py-4 font-medium">August</td>
                              <td className="whitespace-nowrap px-6 py-4">18</td>
                              <td className="whitespace-nowrap px-6 py-4">18</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>

          </dl>
        </div>

      </div>
    </div>
  );
}

export default ResultsExplanation;
