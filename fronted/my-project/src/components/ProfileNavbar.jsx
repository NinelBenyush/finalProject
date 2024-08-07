import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { links,profileLinks,handleFile,account,message } from "../data";
import { FiMessageSquare } from "react-icons/fi";
import { MdAccountCircle } from "react-icons/md";
import { FaHome } from "react-icons/fa";
import { FaFileUpload } from "react-icons/fa";
import { FaFileDownload } from "react-icons/fa";
import { MdInfo } from "react-icons/md";

function ProfileNavbar(){


    const info = profileLinks.filter((link) => link.text!=="Messages")
    const message = profileLinks.filter((link) => link.text!=="Basic Information")
    const filteredLinks = links.filter((link) => link.text !== 'Log in');
    const upload = handleFile.filter((link) => link.text !== 'download file');
    const download =  handleFile.filter((link) => link.text !== 'upload file');
    const m = message.filter((link)=> link.text=="Messages" );
    
    return (
        <nav className="bg-emerald-100">
            <div className="align-element py-4 flex flex-col sm:flex-row sm:gap-x-16 sm:items-center sm:py-8">
                <h2 className="text-3xl font-bold">
                    Order<span className="text-emerald-600">Boost</span>
                </h2>

                          
  <ul className="menu menu-horizontal flex gap-x-3">
  <li>
  {filteredLinks.map((link) => {
                        const { id, href, text} = link;
                        return (
                            <a key={id} href={href} className="tooltip" data-tip="Home">
                            <FaHome className="h-5 w-5" />
                            </a>
                        );
                    })}

  </li>

  <li>
  {account.map((link) => {
                        const { id, href, text} = link;
                        return (
                            <a key={id} href={href} className="tooltip" data-tip="Account">
                            <MdAccountCircle className="h-5 w-5" />
                            </a>
                        );
                    })}
  </li>

  <li>
  {upload.map((link) => {
        const { id, href, text} = link;
             return (
                  <a key={id} href={href} className="tooltip" data-tip="Upload file">
                    <FaFileUpload className="h-5 w-5" />
                     </a>
                    );
    })}
  </li>


  <li>
  {info.map((link) => {
        const { id, href, text} = link;
             return (
                  <a key={id} href={href} className="tooltip" data-tip="Basic info">
                    <MdInfo className="h-5 w-5" />
                     </a>
                    );
    })}
  </li>


  <li>
    {m.map((link) => {
      const {id, href, text} = link;
      return (
        <a key={id} href={href} className='tooltip' data-tip="Messages">
          <FiMessageSquare className="h-5 w-5" />
        </a>
      )
    })}
  </li>
</ul>
 </div>

        </nav>

    )
}

export default ProfileNavbar;