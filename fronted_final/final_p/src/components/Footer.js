import React from "react";
import {FaMailBulk,FaFacebook,FaTwitter} from "react-icons/fa";
import "./footerStyle.css";

function Footer(){
    return (
        <div className="footer">
            <div className="footer-container">
                <div className="left">
                    <div className="phone">
                        <h4><FaMailBulk size={20} style={{color:"fff", marginRight:"2rem"}} />
                            orderboost@gmail.com</h4>
                        
                    </div>
                </div>
                <div className="right">
                    <h4>About us</h4>
                    <p>Each store may face situations of either inventory shortages or excess inventory, both of which can lead to financial losses.</p>
                    <p>Our solution is to predict the amount of inventory that needs to be ordered for the upcoming months.</p>
                    <div className="social">
                      <FaFacebook size={30} style={{color:"fff", marginRight:"1rem"}} />
                      <FaTwitter size={30} style={{color:"fff", marginRight:"1rem"}} />
                    </div>
                </div>

            </div>
        </div>
    )

}

export default Footer;