const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["assets/BuildingReader-27wCQNj3.js","assets/react-CYzKIDNi.js","assets/index-DCjHHh2n.js","assets/createLucideIcon-BggVzj1V.js","assets/jsx-runtime-pGuPoSFy.js","assets/authStore-GPdVY24T.js","assets/index--u1USjrv.css","assets/MobileControls-ElNDDWgv.js"])))=>i.map(i=>d[i]);
import{i as e,n as t,t as n}from"./react-CYzKIDNi.js";import{t as r}from"./jsx-runtime-pGuPoSFy.js";import{Q as i,S as a,_ as o,a as s,b as c,c as l,d as u,f as d,g as f,h as p,i as m,m as h,n as g,o as _,p as v,r as y,s as b,t as x,v as S,x as C,y as w}from"./index-DCjHHh2n.js";var T=e(n(),1),E=i(),D=1e3,O=1001,k=1002,A=1003,j=1004,M=1005,N=1006,P=1007,F=1008,I=1009,ee=1010,te=1011,L=1012,R=1013,ne=1014,re=1015,ie=1016,z=1017,ae=1018,oe=1020,se=35902,ce=35899,le=1021,B=1022,ue=1023,V=1026,de=1027,fe=1028,pe=1029,me=1030,he=1031,ge=1033,_e=33776,ve=33777,ye=33778,be=33779,xe=35840,Se=35841,Ce=35842,we=35843,Te=36196,Ee=37492,H=37496,De=37488,Oe=37489,ke=37490,U=37491,Ae=37808,je=37809,Me=37810,Ne=37811,Pe=37812,Fe=37813,Ie=37814,Le=37815,Re=37816,ze=37817,Be=37818,Ve=37819,He=37820,Ue=37821,We=36492,Ge=36494,Ke=36495,qe=36283,Je=36284,Ye=36285,Xe=36286,Ze=2300,Qe=2301,$e=2302,et=2303,tt=2400,nt=2401,rt=2402,it=3200,at=`srgb`,ot=`srgb-linear`,st=`linear`,ct=`srgb`,lt=7680,ut=35044,dt=2e3;function ft(e){for(let t=e.length-1;t>=0;--t)if(e[t]>=65535)return!0;return!1}function pt(e){return ArrayBuffer.isView(e)&&!(e instanceof DataView)}function mt(e){return document.createElementNS(`http://www.w3.org/1999/xhtml`,e)}function ht(){let e=mt(`canvas`);return e.style.display=`block`,e}var gt={},_t=null;function vt(...e){let t=`THREE.`+e.shift();_t?_t(`log`,t,...e):console.log(t,...e)}function yt(e){let t=e[0];if(typeof t==`string`&&t.startsWith(`TSL:`)){let t=e[1];t&&t.isStackTrace?e[0]+=` `+t.getLocation():e[1]=`Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.`}return e}function W(...e){e=yt(e);let t=`THREE.`+e.shift();if(_t)_t(`warn`,t,...e);else{let n=e[0];n&&n.isStackTrace?console.warn(n.getError(t)):console.warn(t,...e)}}function G(...e){e=yt(e);let t=`THREE.`+e.shift();if(_t)_t(`error`,t,...e);else{let n=e[0];n&&n.isStackTrace?console.error(n.getError(t)):console.error(t,...e)}}function bt(...e){let t=e.join(` `);t in gt||(gt[t]=!0,W(...e))}function xt(e,t,n){return new Promise(function(r,i){function a(){switch(e.clientWaitSync(t,e.SYNC_FLUSH_COMMANDS_BIT,0)){case e.WAIT_FAILED:i();break;case e.TIMEOUT_EXPIRED:setTimeout(a,n);break;default:r()}}setTimeout(a,n)})}var St={0:1,2:6,4:7,3:5,1:0,6:2,7:4,5:3},Ct=class{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});let n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){let n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){let n=this._listeners;if(n===void 0)return;let r=n[e];if(r!==void 0){let e=r.indexOf(t);e!==-1&&r.splice(e,1)}}dispatchEvent(e){let t=this._listeners;if(t===void 0)return;let n=t[e.type];if(n!==void 0){e.target=this;let t=n.slice(0);for(let n=0,r=t.length;n<r;n++)t[n].call(this,e);e.target=null}}},wt=`00.01.02.03.04.05.06.07.08.09.0a.0b.0c.0d.0e.0f.10.11.12.13.14.15.16.17.18.19.1a.1b.1c.1d.1e.1f.20.21.22.23.24.25.26.27.28.29.2a.2b.2c.2d.2e.2f.30.31.32.33.34.35.36.37.38.39.3a.3b.3c.3d.3e.3f.40.41.42.43.44.45.46.47.48.49.4a.4b.4c.4d.4e.4f.50.51.52.53.54.55.56.57.58.59.5a.5b.5c.5d.5e.5f.60.61.62.63.64.65.66.67.68.69.6a.6b.6c.6d.6e.6f.70.71.72.73.74.75.76.77.78.79.7a.7b.7c.7d.7e.7f.80.81.82.83.84.85.86.87.88.89.8a.8b.8c.8d.8e.8f.90.91.92.93.94.95.96.97.98.99.9a.9b.9c.9d.9e.9f.a0.a1.a2.a3.a4.a5.a6.a7.a8.a9.aa.ab.ac.ad.ae.af.b0.b1.b2.b3.b4.b5.b6.b7.b8.b9.ba.bb.bc.bd.be.bf.c0.c1.c2.c3.c4.c5.c6.c7.c8.c9.ca.cb.cc.cd.ce.cf.d0.d1.d2.d3.d4.d5.d6.d7.d8.d9.da.db.dc.dd.de.df.e0.e1.e2.e3.e4.e5.e6.e7.e8.e9.ea.eb.ec.ed.ee.ef.f0.f1.f2.f3.f4.f5.f6.f7.f8.f9.fa.fb.fc.fd.fe.ff`.split(`.`),Tt=Math.PI/180,Et=180/Math.PI;function Dt(){let e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0,r=Math.random()*4294967295|0;return(wt[e&255]+wt[e>>8&255]+wt[e>>16&255]+wt[e>>24&255]+`-`+wt[t&255]+wt[t>>8&255]+`-`+wt[t>>16&15|64]+wt[t>>24&255]+`-`+wt[n&63|128]+wt[n>>8&255]+`-`+wt[n>>16&255]+wt[n>>24&255]+wt[r&255]+wt[r>>8&255]+wt[r>>16&255]+wt[r>>24&255]).toLowerCase()}function K(e,t,n){return Math.max(t,Math.min(n,e))}function Ot(e,t){return(e%t+t)%t}function kt(e,t,n){return(1-n)*e+n*t}function At(e,t){switch(t.constructor){case Float32Array:return e;case Uint32Array:return e/4294967295;case Uint16Array:return e/65535;case Uint8Array:return e/255;case Int32Array:return Math.max(e/2147483647,-1);case Int16Array:return Math.max(e/32767,-1);case Int8Array:return Math.max(e/127,-1);default:throw Error(`THREE.MathUtils: Invalid component type.`)}}function jt(e,t){switch(t.constructor){case Float32Array:return e;case Uint32Array:return Math.round(e*4294967295);case Uint16Array:return Math.round(e*65535);case Uint8Array:return Math.round(e*255);case Int32Array:return Math.round(e*2147483647);case Int16Array:return Math.round(e*32767);case Int8Array:return Math.round(e*127);default:throw Error(`THREE.MathUtils: Invalid component type.`)}}var Mt=class e{static{e.prototype.isVector2=!0}constructor(e=0,t=0){this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw Error(`THREE.Vector2: index is out of range: `+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw Error(`THREE.Vector2: index is out of range: `+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let t=this.x,n=this.y,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6],this.y=r[1]*t+r[4]*n+r[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=K(this.x,e.x,t.x),this.y=K(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=K(this.x,e,t),this.y=K(this.y,e,t),this}clampLength(e,t){let n=this.length();return this.divideScalar(n||1).multiplyScalar(K(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let n=this.dot(e)/t;return Math.acos(K(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){let n=Math.cos(t),r=Math.sin(t),i=this.x-e.x,a=this.y-e.y;return this.x=i*n-a*r+e.x,this.y=i*r+a*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}},Nt=class{constructor(e=0,t=0,n=0,r=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=r}static slerpFlat(e,t,n,r,i,a,o){let s=n[r+0],c=n[r+1],l=n[r+2],u=n[r+3],d=i[a+0],f=i[a+1],p=i[a+2],m=i[a+3];if(u!==m||s!==d||c!==f||l!==p){let e=s*d+c*f+l*p+u*m;e<0&&(d=-d,f=-f,p=-p,m=-m,e=-e);let t=1-o;if(e<.9995){let n=Math.acos(e),r=Math.sin(n);t=Math.sin(t*n)/r,o=Math.sin(o*n)/r,s=s*t+d*o,c=c*t+f*o,l=l*t+p*o,u=u*t+m*o}else{s=s*t+d*o,c=c*t+f*o,l=l*t+p*o,u=u*t+m*o;let e=1/Math.sqrt(s*s+c*c+l*l+u*u);s*=e,c*=e,l*=e,u*=e}}e[t]=s,e[t+1]=c,e[t+2]=l,e[t+3]=u}static multiplyQuaternionsFlat(e,t,n,r,i,a){let o=n[r],s=n[r+1],c=n[r+2],l=n[r+3],u=i[a],d=i[a+1],f=i[a+2],p=i[a+3];return e[t]=o*p+l*u+s*f-c*d,e[t+1]=s*p+l*d+c*u-o*f,e[t+2]=c*p+l*f+o*d-s*u,e[t+3]=l*p-o*u-s*d-c*f,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,r){return this._x=e,this._y=t,this._z=n,this._w=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){let n=e._x,r=e._y,i=e._z,a=e._order,o=Math.cos,s=Math.sin,c=o(n/2),l=o(r/2),u=o(i/2),d=s(n/2),f=s(r/2),p=s(i/2);switch(a){case`XYZ`:this._x=d*l*u+c*f*p,this._y=c*f*u-d*l*p,this._z=c*l*p+d*f*u,this._w=c*l*u-d*f*p;break;case`YXZ`:this._x=d*l*u+c*f*p,this._y=c*f*u-d*l*p,this._z=c*l*p-d*f*u,this._w=c*l*u+d*f*p;break;case`ZXY`:this._x=d*l*u-c*f*p,this._y=c*f*u+d*l*p,this._z=c*l*p+d*f*u,this._w=c*l*u-d*f*p;break;case`ZYX`:this._x=d*l*u-c*f*p,this._y=c*f*u+d*l*p,this._z=c*l*p-d*f*u,this._w=c*l*u+d*f*p;break;case`YZX`:this._x=d*l*u+c*f*p,this._y=c*f*u+d*l*p,this._z=c*l*p-d*f*u,this._w=c*l*u-d*f*p;break;case`XZY`:this._x=d*l*u-c*f*p,this._y=c*f*u-d*l*p,this._z=c*l*p+d*f*u,this._w=c*l*u+d*f*p;break;default:W(`Quaternion: .setFromEuler() encountered an unknown order: `+a)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){let n=t/2,r=Math.sin(n);return this._x=e.x*r,this._y=e.y*r,this._z=e.z*r,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){let t=e.elements,n=t[0],r=t[4],i=t[8],a=t[1],o=t[5],s=t[9],c=t[2],l=t[6],u=t[10],d=n+o+u;if(d>0){let e=.5/Math.sqrt(d+1);this._w=.25/e,this._x=(l-s)*e,this._y=(i-c)*e,this._z=(a-r)*e}else if(n>o&&n>u){let e=2*Math.sqrt(1+n-o-u);this._w=(l-s)/e,this._x=.25*e,this._y=(r+a)/e,this._z=(i+c)/e}else if(o>u){let e=2*Math.sqrt(1+o-n-u);this._w=(i-c)/e,this._x=(r+a)/e,this._y=.25*e,this._z=(s+l)/e}else{let e=2*Math.sqrt(1+u-n-o);this._w=(a-r)/e,this._x=(i+c)/e,this._y=(s+l)/e,this._z=.25*e}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(K(this.dot(e),-1,1)))}rotateTowards(e,t){let n=this.angleTo(e);if(n===0)return this;let r=Math.min(1,t/n);return this.slerp(e,r),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x*=e,this._y*=e,this._z*=e,this._w*=e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){let n=e._x,r=e._y,i=e._z,a=e._w,o=t._x,s=t._y,c=t._z,l=t._w;return this._x=n*l+a*o+r*c-i*s,this._y=r*l+a*s+i*o-n*c,this._z=i*l+a*c+n*s-r*o,this._w=a*l-n*o-r*s-i*c,this._onChangeCallback(),this}slerp(e,t){let n=e._x,r=e._y,i=e._z,a=e._w,o=this.dot(e);o<0&&(n=-n,r=-r,i=-i,a=-a,o=-o);let s=1-t;if(o<.9995){let e=Math.acos(o),c=Math.sin(e);s=Math.sin(s*e)/c,t=Math.sin(t*e)/c,this._x=this._x*s+n*t,this._y=this._y*s+r*t,this._z=this._z*s+i*t,this._w=this._w*s+a*t,this._onChangeCallback()}else this._x=this._x*s+n*t,this._y=this._y*s+r*t,this._z=this._z*s+i*t,this._w=this._w*s+a*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){let e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),r=Math.sqrt(1-n),i=Math.sqrt(n);return this.set(r*Math.sin(e),r*Math.cos(e),i*Math.sin(t),i*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},q=class e{static{e.prototype.isVector3=!0}constructor(e=0,t=0,n=0){this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw Error(`THREE.Vector3: index is out of range: `+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw Error(`THREE.Vector3: index is out of range: `+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Ft.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Ft.setFromAxisAngle(e,t))}applyMatrix3(e){let t=this.x,n=this.y,r=this.z,i=e.elements;return this.x=i[0]*t+i[3]*n+i[6]*r,this.y=i[1]*t+i[4]*n+i[7]*r,this.z=i[2]*t+i[5]*n+i[8]*r,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let t=this.x,n=this.y,r=this.z,i=e.elements,a=1/(i[3]*t+i[7]*n+i[11]*r+i[15]);return this.x=(i[0]*t+i[4]*n+i[8]*r+i[12])*a,this.y=(i[1]*t+i[5]*n+i[9]*r+i[13])*a,this.z=(i[2]*t+i[6]*n+i[10]*r+i[14])*a,this}applyQuaternion(e){let t=this.x,n=this.y,r=this.z,i=e.x,a=e.y,o=e.z,s=e.w,c=2*(a*r-o*n),l=2*(o*t-i*r),u=2*(i*n-a*t);return this.x=t+s*c+a*u-o*l,this.y=n+s*l+o*c-i*u,this.z=r+s*u+i*l-a*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let t=this.x,n=this.y,r=this.z,i=e.elements;return this.x=i[0]*t+i[4]*n+i[8]*r,this.y=i[1]*t+i[5]*n+i[9]*r,this.z=i[2]*t+i[6]*n+i[10]*r,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=K(this.x,e.x,t.x),this.y=K(this.y,e.y,t.y),this.z=K(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=K(this.x,e,t),this.y=K(this.y,e,t),this.z=K(this.z,e,t),this}clampLength(e,t){let n=this.length();return this.divideScalar(n||1).multiplyScalar(K(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){let n=e.x,r=e.y,i=e.z,a=t.x,o=t.y,s=t.z;return this.x=r*s-i*o,this.y=i*a-n*s,this.z=n*o-r*a,this}projectOnVector(e){let t=e.lengthSq();if(t===0)return this.set(0,0,0);let n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Pt.copy(this).projectOnVector(e),this.sub(Pt)}reflect(e){return this.sub(Pt.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let n=this.dot(e)/t;return Math.acos(K(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,n=this.y-e.y,r=this.z-e.z;return t*t+n*n+r*r}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){let r=Math.sin(t)*e;return this.x=r*Math.sin(n),this.y=Math.cos(t)*e,this.z=r*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){let t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),r=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=r,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}},Pt=new q,Ft=new Nt,J=class e{static{e.prototype.isMatrix3=!0}constructor(e,t,n,r,i,a,o,s,c){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,r,i,a,o,s,c)}set(e,t,n,r,i,a,o,s,c){let l=this.elements;return l[0]=e,l[1]=r,l[2]=o,l[3]=t,l[4]=i,l[5]=s,l[6]=n,l[7]=a,l[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let n=e.elements,r=t.elements,i=this.elements,a=n[0],o=n[3],s=n[6],c=n[1],l=n[4],u=n[7],d=n[2],f=n[5],p=n[8],m=r[0],h=r[3],g=r[6],_=r[1],v=r[4],y=r[7],b=r[2],x=r[5],S=r[8];return i[0]=a*m+o*_+s*b,i[3]=a*h+o*v+s*x,i[6]=a*g+o*y+s*S,i[1]=c*m+l*_+u*b,i[4]=c*h+l*v+u*x,i[7]=c*g+l*y+u*S,i[2]=d*m+f*_+p*b,i[5]=d*h+f*v+p*x,i[8]=d*g+f*y+p*S,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){let e=this.elements,t=e[0],n=e[1],r=e[2],i=e[3],a=e[4],o=e[5],s=e[6],c=e[7],l=e[8];return t*a*l-t*o*c-n*i*l+n*o*s+r*i*c-r*a*s}invert(){let e=this.elements,t=e[0],n=e[1],r=e[2],i=e[3],a=e[4],o=e[5],s=e[6],c=e[7],l=e[8],u=l*a-o*c,d=o*s-l*i,f=c*i-a*s,p=t*u+n*d+r*f;if(p===0)return this.set(0,0,0,0,0,0,0,0,0);let m=1/p;return e[0]=u*m,e[1]=(r*c-l*n)*m,e[2]=(o*n-r*a)*m,e[3]=d*m,e[4]=(l*t-r*s)*m,e[5]=(r*i-o*t)*m,e[6]=f*m,e[7]=(n*s-c*t)*m,e[8]=(a*t-n*i)*m,this}transpose(){let e,t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,r,i,a,o){let s=Math.cos(i),c=Math.sin(i);return this.set(n*s,n*c,-n*(s*a+c*o)+a+e,-r*c,r*s,-r*(-c*a+s*o)+o+t,0,0,1),this}scale(e,t){return bt(`Matrix3: .scale() is deprecated. Use .makeScale() instead.`),this.premultiply(It.makeScale(e,t)),this}rotate(e){return bt(`Matrix3: .rotate() is deprecated. Use .makeRotation() instead.`),this.premultiply(It.makeRotation(-e)),this}translate(e,t){return bt(`Matrix3: .translate() is deprecated. Use .makeTranslation() instead.`),this.premultiply(It.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){let t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){let t=this.elements,n=e.elements;for(let e=0;e<9;e++)if(t[e]!==n[e])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){let n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}},It=new J,Lt=new J().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Rt=new J().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function zt(){let e={enabled:!0,workingColorSpace:ot,spaces:{},convert:function(e,t,n){return this.enabled===!1||t===n||!t||!n?e:(this.spaces[t].transfer===`srgb`&&(e.r=Bt(e.r),e.g=Bt(e.g),e.b=Bt(e.b)),this.spaces[t].primaries!==this.spaces[n].primaries&&(e.applyMatrix3(this.spaces[t].toXYZ),e.applyMatrix3(this.spaces[n].fromXYZ)),this.spaces[n].transfer===`srgb`&&(e.r=Vt(e.r),e.g=Vt(e.g),e.b=Vt(e.b)),e)},workingToColorSpace:function(e,t){return this.convert(e,this.workingColorSpace,t)},colorSpaceToWorking:function(e,t){return this.convert(e,t,this.workingColorSpace)},getPrimaries:function(e){return this.spaces[e].primaries},getTransfer:function(e){return e===``?st:this.spaces[e].transfer},getToneMappingMode:function(e){return this.spaces[e].outputColorSpaceConfig.toneMappingMode||`standard`},getLuminanceCoefficients:function(e,t=this.workingColorSpace){return e.fromArray(this.spaces[t].luminanceCoefficients)},define:function(e){Object.assign(this.spaces,e)},_getMatrix:function(e,t,n){return e.copy(this.spaces[t].toXYZ).multiply(this.spaces[n].fromXYZ)},_getDrawingBufferColorSpace:function(e){return this.spaces[e].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(e=this.workingColorSpace){return this.spaces[e].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(t,n){return bt(`ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace().`),e.workingToColorSpace(t,n)},toWorkingColorSpace:function(t,n){return bt(`ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking().`),e.colorSpaceToWorking(t,n)}},t=[.64,.33,.3,.6,.15,.06],n=[.2126,.7152,.0722],r=[.3127,.329];return e.define({[ot]:{primaries:t,whitePoint:r,transfer:st,toXYZ:Lt,fromXYZ:Rt,luminanceCoefficients:n,workingColorSpaceConfig:{unpackColorSpace:at},outputColorSpaceConfig:{drawingBufferColorSpace:at}},[at]:{primaries:t,whitePoint:r,transfer:ct,toXYZ:Lt,fromXYZ:Rt,luminanceCoefficients:n,outputColorSpaceConfig:{drawingBufferColorSpace:at}}}),e}var Y=zt();function Bt(e){return e<.04045?e*.0773993808:(e*.9478672986+.0521327014)**2.4}function Vt(e){return e<.0031308?e*12.92:1.055*e**.41666-.055}var Ht,Ut=class{static getDataURL(e,t=`image/png`){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>`u`)return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{Ht===void 0&&(Ht=mt(`canvas`)),Ht.width=e.width,Ht.height=e.height;let t=Ht.getContext(`2d`);e instanceof ImageData?t.putImageData(e,0,0):t.drawImage(e,0,0,e.width,e.height),n=Ht}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<`u`&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<`u`&&e instanceof HTMLCanvasElement||typeof ImageBitmap<`u`&&e instanceof ImageBitmap){let t=mt(`canvas`);t.width=e.width,t.height=e.height;let n=t.getContext(`2d`);n.drawImage(e,0,0,e.width,e.height);let r=n.getImageData(0,0,e.width,e.height),i=r.data;for(let e=0;e<i.length;e++)i[e]=Bt(i[e]/255)*255;return n.putImageData(r,0,0),t}else if(e.data){let t=e.data.slice(0);for(let e=0;e<t.length;e++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[e]=Math.floor(Bt(t[e]/255)*255):t[e]=Bt(t[e]);return{data:t,width:e.width,height:e.height}}else return W(`ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied.`),e}},Wt=0,Gt=class{constructor(e=null){this.isSource=!0,Object.defineProperty(this,`id`,{value:Wt++}),this.uuid=Dt(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let t=this.data;return typeof HTMLVideoElement<`u`&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<`u`&&t instanceof VideoFrame?e.set(t.displayWidth,t.displayHeight,0):t===null?e.set(0,0,0):e.set(t.width,t.height,t.depth||0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let t=e===void 0||typeof e==`string`;if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let n={uuid:this.uuid,url:``},r=this.data;if(r!==null){let e;if(Array.isArray(r)){e=[];for(let t=0,n=r.length;t<n;t++)r[t].isDataTexture?e.push(Kt(r[t].image)):e.push(Kt(r[t]))}else e=Kt(r);n.url=e}return t||(e.images[this.uuid]=n),n}};function Kt(e){return typeof HTMLImageElement<`u`&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<`u`&&e instanceof HTMLCanvasElement||typeof ImageBitmap<`u`&&e instanceof ImageBitmap?Ut.getDataURL(e):e.data?{data:Array.from(e.data),width:e.width,height:e.height,type:e.data.constructor.name}:(W(`Texture: Unable to serialize Texture.`),{})}var qt=0,Jt=new q,Yt=class e extends Ct{constructor(t=e.DEFAULT_IMAGE,n=e.DEFAULT_MAPPING,r=O,i=O,a=N,o=F,s=ue,c=I,l=e.DEFAULT_ANISOTROPY,u=``){super(),this.isTexture=!0,Object.defineProperty(this,`id`,{value:qt++}),this.uuid=Dt(),this.name=``,this.source=new Gt(t),this.mipmaps=[],this.mapping=n,this.channel=0,this.wrapS=r,this.wrapT=i,this.magFilter=a,this.minFilter=o,this.anisotropy=l,this.format=s,this.internalFormat=null,this.type=c,this.offset=new Mt(0,0),this.repeat=new Mt(1,1),this.center=new Mt(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new J,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(t&&t.depth&&t.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(Jt).x}get height(){return this.source.getSize(Jt).y}get depth(){return this.source.getSize(Jt).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let t in e){let n=e[t];if(n===void 0){W(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}let r=this[t];if(r===void 0){W(`Texture.setValues(): property '${t}' does not exist.`);continue}r&&n&&r.isVector2&&n.isVector2||r&&n&&r.isVector3&&n.isVector3||r&&n&&r.isMatrix3&&n.isMatrix3?r.copy(n):this[t]=n}}toJSON(e){let t=e===void 0||typeof e==`string`;if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let n={metadata:{version:4.7,type:`Texture`,generator:`Texture.toJSON`},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:`dispose`})}transformUv(e){if(this.mapping!==300)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case D:e.x-=Math.floor(e.x);break;case O:e.x=e.x<0?0:1;break;case k:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x-=Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case D:e.y-=Math.floor(e.y);break;case O:e.y=e.y<0?0:1;break;case k:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y-=Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};Yt.DEFAULT_IMAGE=null,Yt.DEFAULT_MAPPING=300,Yt.DEFAULT_ANISOTROPY=1;var Xt=class e{static{e.prototype.isVector4=!0}constructor(e=0,t=0,n=0,r=1){this.x=e,this.y=t,this.z=n,this.w=r}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,r){return this.x=e,this.y=t,this.z=n,this.w=r,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw Error(`THREE.Vector4: index is out of range: `+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw Error(`THREE.Vector4: index is out of range: `+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w===void 0?1:e.w,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let t=this.x,n=this.y,r=this.z,i=this.w,a=e.elements;return this.x=a[0]*t+a[4]*n+a[8]*r+a[12]*i,this.y=a[1]*t+a[5]*n+a[9]*r+a[13]*i,this.z=a[2]*t+a[6]*n+a[10]*r+a[14]*i,this.w=a[3]*t+a[7]*n+a[11]*r+a[15]*i,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,r,i,a=.01,o=.1,s=e.elements,c=s[0],l=s[4],u=s[8],d=s[1],f=s[5],p=s[9],m=s[2],h=s[6],g=s[10];if(Math.abs(l-d)<a&&Math.abs(u-m)<a&&Math.abs(p-h)<a){if(Math.abs(l+d)<o&&Math.abs(u+m)<o&&Math.abs(p+h)<o&&Math.abs(c+f+g-3)<o)return this.set(1,0,0,0),this;t=Math.PI;let e=(c+1)/2,s=(f+1)/2,_=(g+1)/2,v=(l+d)/4,y=(u+m)/4,b=(p+h)/4;return e>s&&e>_?e<a?(n=0,r=.707106781,i=.707106781):(n=Math.sqrt(e),r=v/n,i=y/n):s>_?s<a?(n=.707106781,r=0,i=.707106781):(r=Math.sqrt(s),n=v/r,i=b/r):_<a?(n=.707106781,r=.707106781,i=0):(i=Math.sqrt(_),n=y/i,r=b/i),this.set(n,r,i,t),this}let _=Math.sqrt((h-p)*(h-p)+(u-m)*(u-m)+(d-l)*(d-l));return Math.abs(_)<.001&&(_=1),this.x=(h-p)/_,this.y=(u-m)/_,this.z=(d-l)/_,this.w=Math.acos((c+f+g-1)/2),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=K(this.x,e.x,t.x),this.y=K(this.y,e.y,t.y),this.z=K(this.z,e.z,t.z),this.w=K(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=K(this.x,e,t),this.y=K(this.y,e,t),this.z=K(this.z,e,t),this.w=K(this.w,e,t),this}clampLength(e,t){let n=this.length();return this.divideScalar(n||1).multiplyScalar(K(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}},Zt=class extends Ct{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:N,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Xt(0,0,e,t),this.scissorTest=!1,this.viewport=new Xt(0,0,e,t),this.textures=[];let r=new Yt({width:e,height:t,depth:n.depth}),i=n.count;for(let e=0;e<i;e++)this.textures[e]=r.clone(),this.textures[e].isRenderTargetTexture=!0,this.textures[e].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview,this.useArrayDepthTexture=n.useArrayDepthTexture}_setTextureOptions(e={}){let t={minFilter:N,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let e=0;e<this.textures.length;e++)this.textures[e].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let r=0,i=this.textures.length;r<i;r++)this.textures[r].image.width=e,this.textures[r].image.height=t,this.textures[r].image.depth=n,this.textures[r].isData3DTexture!==!0&&(this.textures[r].isArrayTexture=this.textures[r].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;let n=Object.assign({},e.textures[t].image);this.textures[t].source=new Gt(n)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:`dispose`})}},Qt=class extends Zt{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}},$t=class extends Yt{constructor(e=null,t=1,n=1,r=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:r},this.magFilter=A,this.minFilter=A,this.wrapR=O,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}},en=class extends Yt{constructor(e=null,t=1,n=1,r=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:r},this.magFilter=A,this.minFilter=A,this.wrapR=O,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}},tn=class e{static{e.prototype.isMatrix4=!0}constructor(e,t,n,r,i,a,o,s,c,l,u,d,f,p,m,h){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,r,i,a,o,s,c,l,u,d,f,p,m,h)}set(e,t,n,r,i,a,o,s,c,l,u,d,f,p,m,h){let g=this.elements;return g[0]=e,g[4]=t,g[8]=n,g[12]=r,g[1]=i,g[5]=a,g[9]=o,g[13]=s,g[2]=c,g[6]=l,g[10]=u,g[14]=d,g[3]=f,g[7]=p,g[11]=m,g[15]=h,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new e().fromArray(this.elements)}copy(e){let t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){let t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){let t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return this.determinantAffine()===0?(e.set(1,0,0),t.set(0,1,0),n.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this)}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let t=this.elements,n=e.elements,r=1/nn.setFromMatrixColumn(e,0).length(),i=1/nn.setFromMatrixColumn(e,1).length(),a=1/nn.setFromMatrixColumn(e,2).length();return t[0]=n[0]*r,t[1]=n[1]*r,t[2]=n[2]*r,t[3]=0,t[4]=n[4]*i,t[5]=n[5]*i,t[6]=n[6]*i,t[7]=0,t[8]=n[8]*a,t[9]=n[9]*a,t[10]=n[10]*a,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){let t=this.elements,n=e.x,r=e.y,i=e.z,a=Math.cos(n),o=Math.sin(n),s=Math.cos(r),c=Math.sin(r),l=Math.cos(i),u=Math.sin(i);if(e.order===`XYZ`){let e=a*l,n=a*u,r=o*l,i=o*u;t[0]=s*l,t[4]=-s*u,t[8]=c,t[1]=n+r*c,t[5]=e-i*c,t[9]=-o*s,t[2]=i-e*c,t[6]=r+n*c,t[10]=a*s}else if(e.order===`YXZ`){let e=s*l,n=s*u,r=c*l,i=c*u;t[0]=e+i*o,t[4]=r*o-n,t[8]=a*c,t[1]=a*u,t[5]=a*l,t[9]=-o,t[2]=n*o-r,t[6]=i+e*o,t[10]=a*s}else if(e.order===`ZXY`){let e=s*l,n=s*u,r=c*l,i=c*u;t[0]=e-i*o,t[4]=-a*u,t[8]=r+n*o,t[1]=n+r*o,t[5]=a*l,t[9]=i-e*o,t[2]=-a*c,t[6]=o,t[10]=a*s}else if(e.order===`ZYX`){let e=a*l,n=a*u,r=o*l,i=o*u;t[0]=s*l,t[4]=r*c-n,t[8]=e*c+i,t[1]=s*u,t[5]=i*c+e,t[9]=n*c-r,t[2]=-c,t[6]=o*s,t[10]=a*s}else if(e.order===`YZX`){let e=a*s,n=a*c,r=o*s,i=o*c;t[0]=s*l,t[4]=i-e*u,t[8]=r*u+n,t[1]=u,t[5]=a*l,t[9]=-o*l,t[2]=-c*l,t[6]=n*u+r,t[10]=e-i*u}else if(e.order===`XZY`){let e=a*s,n=a*c,r=o*s,i=o*c;t[0]=s*l,t[4]=-u,t[8]=c*l,t[1]=e*u+i,t[5]=a*l,t[9]=n*u-r,t[2]=r*u-n,t[6]=o*l,t[10]=i*u+e}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(an,e,on)}lookAt(e,t,n){let r=this.elements;return ln.subVectors(e,t),ln.lengthSq()===0&&(ln.z=1),ln.normalize(),sn.crossVectors(n,ln),sn.lengthSq()===0&&(Math.abs(n.z)===1?ln.x+=1e-4:ln.z+=1e-4,ln.normalize(),sn.crossVectors(n,ln)),sn.normalize(),cn.crossVectors(ln,sn),r[0]=sn.x,r[4]=cn.x,r[8]=ln.x,r[1]=sn.y,r[5]=cn.y,r[9]=ln.y,r[2]=sn.z,r[6]=cn.z,r[10]=ln.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let n=e.elements,r=t.elements,i=this.elements,a=n[0],o=n[4],s=n[8],c=n[12],l=n[1],u=n[5],d=n[9],f=n[13],p=n[2],m=n[6],h=n[10],g=n[14],_=n[3],v=n[7],y=n[11],b=n[15],x=r[0],S=r[4],C=r[8],w=r[12],T=r[1],E=r[5],D=r[9],O=r[13],k=r[2],A=r[6],j=r[10],M=r[14],N=r[3],P=r[7],F=r[11],I=r[15];return i[0]=a*x+o*T+s*k+c*N,i[4]=a*S+o*E+s*A+c*P,i[8]=a*C+o*D+s*j+c*F,i[12]=a*w+o*O+s*M+c*I,i[1]=l*x+u*T+d*k+f*N,i[5]=l*S+u*E+d*A+f*P,i[9]=l*C+u*D+d*j+f*F,i[13]=l*w+u*O+d*M+f*I,i[2]=p*x+m*T+h*k+g*N,i[6]=p*S+m*E+h*A+g*P,i[10]=p*C+m*D+h*j+g*F,i[14]=p*w+m*O+h*M+g*I,i[3]=_*x+v*T+y*k+b*N,i[7]=_*S+v*E+y*A+b*P,i[11]=_*C+v*D+y*j+b*F,i[15]=_*w+v*O+y*M+b*I,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){let e=this.elements,t=e[0],n=e[4],r=e[8],i=e[12],a=e[1],o=e[5],s=e[9],c=e[13],l=e[2],u=e[6],d=e[10],f=e[14],p=e[3],m=e[7],h=e[11],g=e[15],_=s*f-c*d,v=o*f-c*u,y=o*d-s*u,b=a*f-c*l,x=a*d-s*l,S=a*u-o*l;return t*(m*_-h*v+g*y)-n*(p*_-h*b+g*x)+r*(p*v-m*b+g*S)-i*(p*y-m*x+h*S)}determinantAffine(){let e=this.elements,t=e[0],n=e[4],r=e[8],i=e[1],a=e[5],o=e[9],s=e[2],c=e[6],l=e[10];return t*(a*l-o*c)-n*(i*l-o*s)+r*(i*c-a*s)}transpose(){let e=this.elements,t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){let r=this.elements;return e.isVector3?(r[12]=e.x,r[13]=e.y,r[14]=e.z):(r[12]=e,r[13]=t,r[14]=n),this}invert(){let e=this.elements,t=e[0],n=e[1],r=e[2],i=e[3],a=e[4],o=e[5],s=e[6],c=e[7],l=e[8],u=e[9],d=e[10],f=e[11],p=e[12],m=e[13],h=e[14],g=e[15],_=t*o-n*a,v=t*s-r*a,y=t*c-i*a,b=n*s-r*o,x=n*c-i*o,S=r*c-i*s,C=l*m-u*p,w=l*h-d*p,T=l*g-f*p,E=u*h-d*m,D=u*g-f*m,O=d*g-f*h,k=_*O-v*D+y*E+b*T-x*w+S*C;if(k===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let A=1/k;return e[0]=(o*O-s*D+c*E)*A,e[1]=(r*D-n*O-i*E)*A,e[2]=(m*S-h*x+g*b)*A,e[3]=(d*x-u*S-f*b)*A,e[4]=(s*T-a*O-c*w)*A,e[5]=(t*O-r*T+i*w)*A,e[6]=(h*y-p*S-g*v)*A,e[7]=(l*S-d*y+f*v)*A,e[8]=(a*D-o*T+c*C)*A,e[9]=(n*T-t*D-i*C)*A,e[10]=(p*x-m*y+g*_)*A,e[11]=(u*y-l*x-f*_)*A,e[12]=(o*w-a*E-s*C)*A,e[13]=(t*E-n*w+r*C)*A,e[14]=(m*v-p*b-h*_)*A,e[15]=(l*b-u*v+d*_)*A,this}scale(e){let t=this.elements,n=e.x,r=e.y,i=e.z;return t[0]*=n,t[4]*=r,t[8]*=i,t[1]*=n,t[5]*=r,t[9]*=i,t[2]*=n,t[6]*=r,t[10]*=i,t[3]*=n,t[7]*=r,t[11]*=i,this}getMaxScaleOnAxis(){let e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],r=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,r))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){let t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){let t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){let t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){let n=Math.cos(t),r=Math.sin(t),i=1-n,a=e.x,o=e.y,s=e.z,c=i*a,l=i*o;return this.set(c*a+n,c*o-r*s,c*s+r*o,0,c*o+r*s,l*o+n,l*s-r*a,0,c*s-r*o,l*s+r*a,i*s*s+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,r,i,a){return this.set(1,n,i,0,e,1,a,0,t,r,1,0,0,0,0,1),this}compose(e,t,n){let r=this.elements,i=t._x,a=t._y,o=t._z,s=t._w,c=i+i,l=a+a,u=o+o,d=i*c,f=i*l,p=i*u,m=a*l,h=a*u,g=o*u,_=s*c,v=s*l,y=s*u,b=n.x,x=n.y,S=n.z;return r[0]=(1-(m+g))*b,r[1]=(f+y)*b,r[2]=(p-v)*b,r[3]=0,r[4]=(f-y)*x,r[5]=(1-(d+g))*x,r[6]=(h+_)*x,r[7]=0,r[8]=(p+v)*S,r[9]=(h-_)*S,r[10]=(1-(d+m))*S,r[11]=0,r[12]=e.x,r[13]=e.y,r[14]=e.z,r[15]=1,this}decompose(e,t,n){let r=this.elements;e.x=r[12],e.y=r[13],e.z=r[14];let i=this.determinantAffine();if(i===0)return n.set(1,1,1),t.identity(),this;let a=nn.set(r[0],r[1],r[2]).length(),o=nn.set(r[4],r[5],r[6]).length(),s=nn.set(r[8],r[9],r[10]).length();i<0&&(a=-a),rn.copy(this);let c=1/a,l=1/o,u=1/s;return rn.elements[0]*=c,rn.elements[1]*=c,rn.elements[2]*=c,rn.elements[4]*=l,rn.elements[5]*=l,rn.elements[6]*=l,rn.elements[8]*=u,rn.elements[9]*=u,rn.elements[10]*=u,t.setFromRotationMatrix(rn),n.x=a,n.y=o,n.z=s,this}makePerspective(e,t,n,r,i,a,o=dt,s=!1){let c=this.elements,l=2*i/(t-e),u=2*i/(n-r),d=(t+e)/(t-e),f=(n+r)/(n-r),p,m;if(s)p=i/(a-i),m=a*i/(a-i);else if(o===2e3)p=-(a+i)/(a-i),m=-2*a*i/(a-i);else if(o===2001)p=-a/(a-i),m=-a*i/(a-i);else throw Error(`THREE.Matrix4.makePerspective(): Invalid coordinate system: `+o);return c[0]=l,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=u,c[9]=f,c[13]=0,c[2]=0,c[6]=0,c[10]=p,c[14]=m,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,r,i,a,o=dt,s=!1){let c=this.elements,l=2/(t-e),u=2/(n-r),d=-(t+e)/(t-e),f=-(n+r)/(n-r),p,m;if(s)p=1/(a-i),m=a/(a-i);else if(o===2e3)p=-2/(a-i),m=-(a+i)/(a-i);else if(o===2001)p=-1/(a-i),m=-i/(a-i);else throw Error(`THREE.Matrix4.makeOrthographic(): Invalid coordinate system: `+o);return c[0]=l,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=u,c[9]=0,c[13]=f,c[2]=0,c[6]=0,c[10]=p,c[14]=m,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){let t=this.elements,n=e.elements;for(let e=0;e<16;e++)if(t[e]!==n[e])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){let n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}},nn=new q,rn=new tn,an=new q(0,0,0),on=new q(1,1,1),sn=new q,cn=new q,ln=new q,un=new tn,dn=new Nt,fn=class e{constructor(t=0,n=0,r=0,i=e.DEFAULT_ORDER){this.isEuler=!0,this._x=t,this._y=n,this._z=r,this._order=i}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,r=this._order){return this._x=e,this._y=t,this._z=n,this._order=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){let r=e.elements,i=r[0],a=r[4],o=r[8],s=r[1],c=r[5],l=r[9],u=r[2],d=r[6],f=r[10];switch(t){case`XYZ`:this._y=Math.asin(K(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-l,f),this._z=Math.atan2(-a,i)):(this._x=Math.atan2(d,c),this._z=0);break;case`YXZ`:this._x=Math.asin(-K(l,-1,1)),Math.abs(l)<.9999999?(this._y=Math.atan2(o,f),this._z=Math.atan2(s,c)):(this._y=Math.atan2(-u,i),this._z=0);break;case`ZXY`:this._x=Math.asin(K(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-u,f),this._z=Math.atan2(-a,c)):(this._y=0,this._z=Math.atan2(s,i));break;case`ZYX`:this._y=Math.asin(-K(u,-1,1)),Math.abs(u)<.9999999?(this._x=Math.atan2(d,f),this._z=Math.atan2(s,i)):(this._x=0,this._z=Math.atan2(-a,c));break;case`YZX`:this._z=Math.asin(K(s,-1,1)),Math.abs(s)<.9999999?(this._x=Math.atan2(-l,c),this._y=Math.atan2(-u,i)):(this._x=0,this._y=Math.atan2(o,f));break;case`XZY`:this._z=Math.asin(-K(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(o,i)):(this._x=Math.atan2(-l,f),this._y=0);break;default:W(`Euler: .setFromRotationMatrix() encountered an unknown order: `+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return un.makeRotationFromQuaternion(e),this.setFromRotationMatrix(un,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return dn.setFromEuler(this),this.setFromQuaternion(dn,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};fn.DEFAULT_ORDER=`XYZ`;var pn=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!=0}},mn=0,hn=new q,gn=new Nt,_n=new tn,vn=new q,yn=new q,bn=new q,xn=new Nt,Sn=new q(1,0,0),Cn=new q(0,1,0),wn=new q(0,0,1),Tn={type:`added`},En={type:`removed`},Dn={type:`childadded`,child:null},On={type:`childremoved`,child:null},kn=class e extends Ct{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,`id`,{value:mn++}),this.uuid=Dt(),this.name=``,this.type=`Object3D`,this.parent=null,this.children=[],this.up=e.DEFAULT_UP.clone();let t=new q,n=new fn,r=new Nt,i=new q(1,1,1);function a(){r.setFromEuler(n,!1)}function o(){n.setFromQuaternion(r,void 0,!1)}n._onChange(a),r._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:n},quaternion:{configurable:!0,enumerable:!0,value:r},scale:{configurable:!0,enumerable:!0,value:i},modelViewMatrix:{value:new tn},normalMatrix:{value:new J}}),this.matrix=new tn,this.matrixWorld=new tn,this.matrixAutoUpdate=e.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=e.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new pn,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return gn.setFromAxisAngle(e,t),this.quaternion.multiply(gn),this}rotateOnWorldAxis(e,t){return gn.setFromAxisAngle(e,t),this.quaternion.premultiply(gn),this}rotateX(e){return this.rotateOnAxis(Sn,e)}rotateY(e){return this.rotateOnAxis(Cn,e)}rotateZ(e){return this.rotateOnAxis(wn,e)}translateOnAxis(e,t){return hn.copy(e).applyQuaternion(this.quaternion),this.position.add(hn.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Sn,e)}translateY(e){return this.translateOnAxis(Cn,e)}translateZ(e){return this.translateOnAxis(wn,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(_n.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?vn.copy(e):vn.set(e,t,n);let r=this.parent;this.updateWorldMatrix(!0,!1),yn.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?_n.lookAt(yn,vn,this.up):_n.lookAt(vn,yn,this.up),this.quaternion.setFromRotationMatrix(_n),r&&(_n.extractRotation(r.matrixWorld),gn.setFromRotationMatrix(_n),this.quaternion.premultiply(gn.invert()))}add(e){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.add(arguments[e]);return this}return e===this?(G(`Object3D.add: object can't be added as a child of itself.`,e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Tn),Dn.child=e,this.dispatchEvent(Dn),Dn.child=null):G(`Object3D.add: object not an instance of THREE.Object3D.`,e),this)}remove(e){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.remove(arguments[e]);return this}let t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(En),On.child=e,this.dispatchEvent(On),On.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),_n.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),_n.multiply(e.parent.matrixWorld)),e.applyMatrix4(_n),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Tn),Dn.child=e,this.dispatchEvent(Dn),Dn.child=null,this}getObjectById(e){return this.getObjectByProperty(`id`,e)}getObjectByName(e){return this.getObjectByProperty(`name`,e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,r=this.children.length;n<r;n++){let r=this.children[n].getObjectByProperty(e,t);if(r!==void 0)return r}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);let r=this.children;for(let i=0,a=r.length;i<a;i++)r[i].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(yn,e,bn),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(yn,xn,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);let t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].traverseVisible(e)}traverseAncestors(e){let t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let t=e.x,n=e.y,r=e.z,i=this.matrix.elements;i[12]+=t-i[0]*t-i[4]*n-i[8]*r,i[13]+=n-i[1]*t-i[5]*n-i[9]*r,i[14]+=r-i[2]*t-i[6]*n-i[10]*r}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t,n=!1){let r=this.parent;if(e===!0&&r!==null&&r.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||n)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,n=!0),t===!0){let e=this.children;for(let t=0,r=e.length;t<r;t++)e[t].updateWorldMatrix(!1,!0,n)}}toJSON(e){let t=e===void 0||typeof e==`string`,n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:`Object`,generator:`Object3D.toJSON`});let r={};r.uuid=this.uuid,r.type=this.type,this.name!==``&&(r.name=this.name),this.castShadow===!0&&(r.castShadow=!0),this.receiveShadow===!0&&(r.receiveShadow=!0),this.visible===!1&&(r.visible=!1),this.frustumCulled===!1&&(r.frustumCulled=!1),this.renderOrder!==0&&(r.renderOrder=this.renderOrder),this.static!==!1&&(r.static=this.static),Object.keys(this.userData).length>0&&(r.userData=this.userData),r.layers=this.layers.mask,r.matrix=this.matrix.toArray(),r.up=this.up.toArray(),this.pivot!==null&&(r.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(r.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(r.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(r.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(r.type=`InstancedMesh`,r.count=this.count,r.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(r.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(r.type=`BatchedMesh`,r.perObjectFrustumCulled=this.perObjectFrustumCulled,r.sortObjects=this.sortObjects,r.drawRanges=this._drawRanges,r.reservedRanges=this._reservedRanges,r.geometryInfo=this._geometryInfo.map(e=>({...e,boundingBox:e.boundingBox?e.boundingBox.toJSON():void 0,boundingSphere:e.boundingSphere?e.boundingSphere.toJSON():void 0})),r.instanceInfo=this._instanceInfo.map(e=>({...e})),r.availableInstanceIds=this._availableInstanceIds.slice(),r.availableGeometryIds=this._availableGeometryIds.slice(),r.nextIndexStart=this._nextIndexStart,r.nextVertexStart=this._nextVertexStart,r.geometryCount=this._geometryCount,r.maxInstanceCount=this._maxInstanceCount,r.maxVertexCount=this._maxVertexCount,r.maxIndexCount=this._maxIndexCount,r.geometryInitialized=this._geometryInitialized,r.matricesTexture=this._matricesTexture.toJSON(e),r.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(r.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(r.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(r.boundingBox=this.boundingBox.toJSON()));function i(t,n){return t[n.uuid]===void 0&&(t[n.uuid]=n.toJSON(e)),n.uuid}if(this.isScene)this.background&&(this.background.isColor?r.background=this.background.toJSON():this.background.isTexture&&(r.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(r.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){r.geometry=i(e.geometries,this.geometry);let t=this.geometry.parameters;if(t!==void 0&&t.shapes!==void 0){let n=t.shapes;if(Array.isArray(n))for(let t=0,r=n.length;t<r;t++){let r=n[t];i(e.shapes,r)}else i(e.shapes,n)}}if(this.isSkinnedMesh&&(r.bindMode=this.bindMode,r.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(i(e.skeletons,this.skeleton),r.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let t=[];for(let n=0,r=this.material.length;n<r;n++)t.push(i(e.materials,this.material[n]));r.material=t}else r.material=i(e.materials,this.material);if(this.children.length>0){r.children=[];for(let t=0;t<this.children.length;t++)r.children.push(this.children[t].toJSON(e).object)}if(this.animations.length>0){r.animations=[];for(let t=0;t<this.animations.length;t++){let n=this.animations[t];r.animations.push(i(e.animations,n))}}if(t){let t=a(e.geometries),r=a(e.materials),i=a(e.textures),o=a(e.images),s=a(e.shapes),c=a(e.skeletons),l=a(e.animations),u=a(e.nodes);t.length>0&&(n.geometries=t),r.length>0&&(n.materials=r),i.length>0&&(n.textures=i),o.length>0&&(n.images=o),s.length>0&&(n.shapes=s),c.length>0&&(n.skeletons=c),l.length>0&&(n.animations=l),u.length>0&&(n.nodes=u)}return n.object=r,n;function a(e){let t=[];for(let n in e){let r=e[n];delete r.metadata,t.push(r)}return t}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot===null?null:e.pivot.clone(),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let t=0;t<e.children.length;t++){let n=e.children[t];this.add(n.clone())}return this}};kn.DEFAULT_UP=new q(0,1,0),kn.DEFAULT_MATRIX_AUTO_UPDATE=!0,kn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var An=class extends kn{constructor(){super(),this.isGroup=!0,this.type=`Group`}},jn={type:`move`},Mn=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new An,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new An,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new q,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new q),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new An,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new q,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new q,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let t=this._hand;if(t)for(let n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:`connected`,data:e}),this}disconnect(e){return this.dispatchEvent({type:`disconnected`,data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let r=null,i=null,a=null,o=this._targetRay,s=this._grip,c=this._hand;if(e&&t.session.visibilityState!==`visible-blurred`){if(c&&e.hand){a=!0;for(let r of e.hand.values()){let e=t.getJointPose(r,n),i=this._getHandJoint(c,r);e!==null&&(i.matrix.fromArray(e.transform.matrix),i.matrix.decompose(i.position,i.rotation,i.scale),i.matrixWorldNeedsUpdate=!0,i.jointRadius=e.radius),i.visible=e!==null}let r=c.joints[`index-finger-tip`],i=c.joints[`thumb-tip`],o=r.position.distanceTo(i.position),s=.02,l=.005;c.inputState.pinching&&o>s+l?(c.inputState.pinching=!1,this.dispatchEvent({type:`pinchend`,handedness:e.handedness,target:this})):!c.inputState.pinching&&o<=s-l&&(c.inputState.pinching=!0,this.dispatchEvent({type:`pinchstart`,handedness:e.handedness,target:this}))}else s!==null&&e.gripSpace&&(i=t.getPose(e.gripSpace,n),i!==null&&(s.matrix.fromArray(i.transform.matrix),s.matrix.decompose(s.position,s.rotation,s.scale),s.matrixWorldNeedsUpdate=!0,i.linearVelocity?(s.hasLinearVelocity=!0,s.linearVelocity.copy(i.linearVelocity)):s.hasLinearVelocity=!1,i.angularVelocity?(s.hasAngularVelocity=!0,s.angularVelocity.copy(i.angularVelocity)):s.hasAngularVelocity=!1,s.eventsEnabled&&s.dispatchEvent({type:`gripUpdated`,data:e,target:this})));o!==null&&(r=t.getPose(e.targetRaySpace,n),r===null&&i!==null&&(r=i),r!==null&&(o.matrix.fromArray(r.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,r.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(r.linearVelocity)):o.hasLinearVelocity=!1,r.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(r.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(jn)))}return o!==null&&(o.visible=r!==null),s!==null&&(s.visible=i!==null),c!==null&&(c.visible=a!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){let n=new An;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}},Nn={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Pn={h:0,s:0,l:0},Fn={h:0,s:0,l:0};function In(e,t,n){return n<0&&(n+=1),n>1&&--n,n<1/6?e+(t-e)*6*n:n<1/2?t:n<2/3?e+(t-e)*6*(2/3-n):e}var Ln=class{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){let t=e;t&&t.isColor?this.copy(t):typeof t==`number`?this.setHex(t):typeof t==`string`&&this.setStyle(t)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=at){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,Y.colorSpaceToWorking(this,t),this}setRGB(e,t,n,r=Y.workingColorSpace){return this.r=e,this.g=t,this.b=n,Y.colorSpaceToWorking(this,r),this}setHSL(e,t,n,r=Y.workingColorSpace){if(e=Ot(e,1),t=K(t,0,1),n=K(n,0,1),t===0)this.r=this.g=this.b=n;else{let r=n<=.5?n*(1+t):n+t-n*t,i=2*n-r;this.r=In(i,r,e+1/3),this.g=In(i,r,e),this.b=In(i,r,e-1/3)}return Y.colorSpaceToWorking(this,r),this}setStyle(e,t=at){function n(t){t!==void 0&&parseFloat(t)<1&&W(`Color: Alpha component of `+e+` will be ignored.`)}let r;if(r=/^(\w+)\(([^\)]*)\)/.exec(e)){let i,a=r[1],o=r[2];switch(a){case`rgb`:case`rgba`:if(i=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(i[4]),this.setRGB(Math.min(255,parseInt(i[1],10))/255,Math.min(255,parseInt(i[2],10))/255,Math.min(255,parseInt(i[3],10))/255,t);if(i=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(i[4]),this.setRGB(Math.min(100,parseInt(i[1],10))/100,Math.min(100,parseInt(i[2],10))/100,Math.min(100,parseInt(i[3],10))/100,t);break;case`hsl`:case`hsla`:if(i=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(i[4]),this.setHSL(parseFloat(i[1])/360,parseFloat(i[2])/100,parseFloat(i[3])/100,t);break;default:W(`Color: Unknown color model `+e)}}else if(r=/^\#([A-Fa-f\d]+)$/.exec(e)){let n=r[1],i=n.length;if(i===3)return this.setRGB(parseInt(n.charAt(0),16)/15,parseInt(n.charAt(1),16)/15,parseInt(n.charAt(2),16)/15,t);if(i===6)return this.setHex(parseInt(n,16),t);W(`Color: Invalid hex color `+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=at){let n=Nn[e.toLowerCase()];return n===void 0?W(`Color: Unknown color `+e):this.setHex(n,t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Bt(e.r),this.g=Bt(e.g),this.b=Bt(e.b),this}copyLinearToSRGB(e){return this.r=Vt(e.r),this.g=Vt(e.g),this.b=Vt(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=at){return Y.workingToColorSpace(Rn.copy(this),e),Math.round(K(Rn.r*255,0,255))*65536+Math.round(K(Rn.g*255,0,255))*256+Math.round(K(Rn.b*255,0,255))}getHexString(e=at){return(`000000`+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=Y.workingColorSpace){Y.workingToColorSpace(Rn.copy(this),t);let n=Rn.r,r=Rn.g,i=Rn.b,a=Math.max(n,r,i),o=Math.min(n,r,i),s,c,l=(o+a)/2;if(o===a)s=0,c=0;else{let e=a-o;switch(c=l<=.5?e/(a+o):e/(2-a-o),a){case n:s=(r-i)/e+(r<i?6:0);break;case r:s=(i-n)/e+2;break;case i:s=(n-r)/e+4;break}s/=6}return e.h=s,e.s=c,e.l=l,e}getRGB(e,t=Y.workingColorSpace){return Y.workingToColorSpace(Rn.copy(this),t),e.r=Rn.r,e.g=Rn.g,e.b=Rn.b,e}getStyle(e=at){Y.workingToColorSpace(Rn.copy(this),e);let t=Rn.r,n=Rn.g,r=Rn.b;return e===`srgb`?`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(r*255)})`:`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${r.toFixed(3)})`}offsetHSL(e,t,n){return this.getHSL(Pn),this.setHSL(Pn.h+e,Pn.s+t,Pn.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(Pn),e.getHSL(Fn);let n=kt(Pn.h,Fn.h,t),r=kt(Pn.s,Fn.s,t),i=kt(Pn.l,Fn.l,t);return this.setHSL(n,r,i),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let t=this.r,n=this.g,r=this.b,i=e.elements;return this.r=i[0]*t+i[3]*n+i[6]*r,this.g=i[1]*t+i[4]*n+i[7]*r,this.b=i[2]*t+i[5]*n+i[8]*r,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},Rn=new Ln;Ln.NAMES=Nn;var zn=class e{constructor(e,t=1,n=1e3){this.isFog=!0,this.name=``,this.color=new Ln(e),this.near=t,this.far=n}clone(){return new e(this.color,this.near,this.far)}toJSON(){return{type:`Fog`,name:this.name,color:this.color.getHex(),near:this.near,far:this.far}}},Bn=class extends kn{constructor(){super(),this.isScene=!0,this.type=`Scene`,this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new fn,this.environmentIntensity=1,this.environmentRotation=new fn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<`u`&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent(`observe`,{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}},Vn=new q,Hn=new q,Un=new q,Wn=new q,Gn=new q,Kn=new q,qn=new q,Jn=new q,Yn=new q,Xn=new q,Zn=new Xt,Qn=new Xt,$n=new Xt,er=class e{constructor(e=new q,t=new q,n=new q){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,r){r.subVectors(n,t),Vn.subVectors(e,t),r.cross(Vn);let i=r.lengthSq();return i>0?r.multiplyScalar(1/Math.sqrt(i)):r.set(0,0,0)}static getBarycoord(e,t,n,r,i){Vn.subVectors(r,t),Hn.subVectors(n,t),Un.subVectors(e,t);let a=Vn.dot(Vn),o=Vn.dot(Hn),s=Vn.dot(Un),c=Hn.dot(Hn),l=Hn.dot(Un),u=a*c-o*o;if(u===0)return i.set(0,0,0),null;let d=1/u,f=(c*s-o*l)*d,p=(a*l-o*s)*d;return i.set(1-f-p,p,f)}static containsPoint(e,t,n,r){return this.getBarycoord(e,t,n,r,Wn)===null?!1:Wn.x>=0&&Wn.y>=0&&Wn.x+Wn.y<=1}static getInterpolation(e,t,n,r,i,a,o,s){return this.getBarycoord(e,t,n,r,Wn)===null?(s.x=0,s.y=0,`z`in s&&(s.z=0),`w`in s&&(s.w=0),null):(s.setScalar(0),s.addScaledVector(i,Wn.x),s.addScaledVector(a,Wn.y),s.addScaledVector(o,Wn.z),s)}static getInterpolatedAttribute(e,t,n,r,i,a){return Zn.setScalar(0),Qn.setScalar(0),$n.setScalar(0),Zn.fromBufferAttribute(e,t),Qn.fromBufferAttribute(e,n),$n.fromBufferAttribute(e,r),a.setScalar(0),a.addScaledVector(Zn,i.x),a.addScaledVector(Qn,i.y),a.addScaledVector($n,i.z),a}static isFrontFacing(e,t,n,r){return Vn.subVectors(n,t),Hn.subVectors(e,t),Vn.cross(Hn).dot(r)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,r){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[r]),this}setFromAttributeAndIndices(e,t,n,r){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,r),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Vn.subVectors(this.c,this.b),Hn.subVectors(this.a,this.b),Vn.cross(Hn).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return e.getNormal(this.a,this.b,this.c,t)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,n){return e.getBarycoord(t,this.a,this.b,this.c,n)}getInterpolation(t,n,r,i,a){return e.getInterpolation(t,this.a,this.b,this.c,n,r,i,a)}containsPoint(t){return e.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return e.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){let n=this.a,r=this.b,i=this.c,a,o;Gn.subVectors(r,n),Kn.subVectors(i,n),Jn.subVectors(e,n);let s=Gn.dot(Jn),c=Kn.dot(Jn);if(s<=0&&c<=0)return t.copy(n);Yn.subVectors(e,r);let l=Gn.dot(Yn),u=Kn.dot(Yn);if(l>=0&&u<=l)return t.copy(r);let d=s*u-l*c;if(d<=0&&s>=0&&l<=0)return a=s/(s-l),t.copy(n).addScaledVector(Gn,a);Xn.subVectors(e,i);let f=Gn.dot(Xn),p=Kn.dot(Xn);if(p>=0&&f<=p)return t.copy(i);let m=f*c-s*p;if(m<=0&&c>=0&&p<=0)return o=c/(c-p),t.copy(n).addScaledVector(Kn,o);let h=l*p-f*u;if(h<=0&&u-l>=0&&f-p>=0)return qn.subVectors(i,r),o=(u-l)/(u-l+(f-p)),t.copy(r).addScaledVector(qn,o);let g=1/(h+m+d);return a=m*g,o=d*g,t.copy(n).addScaledVector(Gn,a).addScaledVector(Kn,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},tr=class{constructor(e=new q(1/0,1/0,1/0),t=new q(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(rr.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(rr.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){let n=rr.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);let n=e.geometry;if(n!==void 0){let r=n.getAttribute(`position`);if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let t=0,n=r.count;t<n;t++)e.isMesh===!0?e.getVertexPosition(t,rr):rr.fromBufferAttribute(r,t),rr.applyMatrix4(e.matrixWorld),this.expandByPoint(rr);else e.boundingBox===void 0?(n.boundingBox===null&&n.computeBoundingBox(),ir.copy(n.boundingBox)):(e.boundingBox===null&&e.computeBoundingBox(),ir.copy(e.boundingBox)),ir.applyMatrix4(e.matrixWorld),this.union(ir)}let r=e.children;for(let e=0,n=r.length;e<n;e++)this.expandByObject(r[e],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,rr),rr.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(dr),fr.subVectors(this.max,dr),ar.subVectors(e.a,dr),or.subVectors(e.b,dr),sr.subVectors(e.c,dr),cr.subVectors(or,ar),lr.subVectors(sr,or),ur.subVectors(ar,sr);let t=[0,-cr.z,cr.y,0,-lr.z,lr.y,0,-ur.z,ur.y,cr.z,0,-cr.x,lr.z,0,-lr.x,ur.z,0,-ur.x,-cr.y,cr.x,0,-lr.y,lr.x,0,-ur.y,ur.x,0];return!hr(t,ar,or,sr,fr)||(t=[1,0,0,0,1,0,0,0,1],!hr(t,ar,or,sr,fr))?!1:(pr.crossVectors(cr,lr),t=[pr.x,pr.y,pr.z],hr(t,ar,or,sr,fr))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,rr).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(rr).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(nr[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),nr[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),nr[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),nr[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),nr[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),nr[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),nr[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),nr[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(nr),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},nr=[new q,new q,new q,new q,new q,new q,new q,new q],rr=new q,ir=new tr,ar=new q,or=new q,sr=new q,cr=new q,lr=new q,ur=new q,dr=new q,fr=new q,pr=new q,mr=new q;function hr(e,t,n,r,i){for(let a=0,o=e.length-3;a<=o;a+=3){mr.fromArray(e,a);let o=i.x*Math.abs(mr.x)+i.y*Math.abs(mr.y)+i.z*Math.abs(mr.z),s=t.dot(mr),c=n.dot(mr),l=r.dot(mr);if(Math.max(-Math.max(s,c,l),Math.min(s,c,l))>o)return!1}return!0}var gr=new q,_r=new Mt,vr=0,yr=class extends Ct{constructor(e,t,n=!1){if(super(),Array.isArray(e))throw TypeError(`THREE.BufferAttribute: array should be a Typed Array.`);this.isBufferAttribute=!0,Object.defineProperty(this,`id`,{value:vr++}),this.name=``,this.array=e,this.itemSize=t,this.count=e===void 0?0:e.length/t,this.normalized=n,this.usage=ut,this.updateRanges=[],this.gpuType=re,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let r=0,i=this.itemSize;r<i;r++)this.array[e+r]=t.array[n+r];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)_r.fromBufferAttribute(this,t),_r.applyMatrix3(e),this.setXY(t,_r.x,_r.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)gr.fromBufferAttribute(this,t),gr.applyMatrix3(e),this.setXYZ(t,gr.x,gr.y,gr.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)gr.fromBufferAttribute(this,t),gr.applyMatrix4(e),this.setXYZ(t,gr.x,gr.y,gr.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)gr.fromBufferAttribute(this,t),gr.applyNormalMatrix(e),this.setXYZ(t,gr.x,gr.y,gr.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)gr.fromBufferAttribute(this,t),gr.transformDirection(e),this.setXYZ(t,gr.x,gr.y,gr.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=At(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=jt(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=At(t,this.array)),t}setX(e,t){return this.normalized&&(t=jt(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=At(t,this.array)),t}setY(e,t){return this.normalized&&(t=jt(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=At(t,this.array)),t}setZ(e,t){return this.normalized&&(t=jt(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=At(t,this.array)),t}setW(e,t){return this.normalized&&(t=jt(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=jt(t,this.array),n=jt(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,r){return e*=this.itemSize,this.normalized&&(t=jt(t,this.array),n=jt(n,this.array),r=jt(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=r,this}setXYZW(e,t,n,r,i){return e*=this.itemSize,this.normalized&&(t=jt(t,this.array),n=jt(n,this.array),r=jt(r,this.array),i=jt(i,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=r,this.array[e+3]=i,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==``&&(e.name=this.name),this.usage!==35044&&(e.usage=this.usage),e}dispose(){this.dispatchEvent({type:`dispose`})}},br=class extends yr{constructor(e,t,n){super(new Uint16Array(e),t,n)}},xr=class extends yr{constructor(e,t,n){super(new Uint32Array(e),t,n)}},Sr=class extends yr{constructor(e,t,n){super(new Float32Array(e),t,n)}},Cr=new tr,wr=new q,Tr=new q,Er=class{constructor(e=new q,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){let n=this.center;t===void 0?Cr.setFromPoints(e).getCenter(n):n.copy(t);let r=0;for(let t=0,i=e.length;t<i;t++)r=Math.max(r,n.distanceToSquared(e[t]));return this.radius=Math.sqrt(r),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){let n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius*=e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;wr.subVectors(e,this.center);let t=wr.lengthSq();if(t>this.radius*this.radius){let e=Math.sqrt(t),n=(e-this.radius)*.5;this.center.addScaledVector(wr,n/e),this.radius+=n}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Tr.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(wr.copy(e.center).add(Tr)),this.expandByPoint(wr.copy(e.center).sub(Tr))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},Dr=0,Or=new tn,kr=new kn,Ar=new q,jr=new tr,Mr=new tr,Nr=new q,Pr=class e extends Ct{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,`id`,{value:Dr++}),this.uuid=Dt(),this.name=``,this.type=`BufferGeometry`,this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(ft(e)?xr:br)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){let t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);let n=this.attributes.normal;if(n!==void 0){let t=new J().getNormalMatrix(e);n.applyNormalMatrix(t),n.needsUpdate=!0}let r=this.attributes.tangent;return r!==void 0&&(r.transformDirection(e),r.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return Or.makeRotationFromQuaternion(e),this.applyMatrix4(Or),this}rotateX(e){return Or.makeRotationX(e),this.applyMatrix4(Or),this}rotateY(e){return Or.makeRotationY(e),this.applyMatrix4(Or),this}rotateZ(e){return Or.makeRotationZ(e),this.applyMatrix4(Or),this}translate(e,t,n){return Or.makeTranslation(e,t,n),this.applyMatrix4(Or),this}scale(e,t,n){return Or.makeScale(e,t,n),this.applyMatrix4(Or),this}lookAt(e){return kr.lookAt(e),kr.updateMatrix(),this.applyMatrix4(kr.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Ar).negate(),this.translate(Ar.x,Ar.y,Ar.z),this}setFromPoints(e){let t=this.getAttribute(`position`);if(t===void 0){let t=[];for(let n=0,r=e.length;n<r;n++){let r=e[n];t.push(r.x,r.y,r.z||0)}this.setAttribute(`position`,new Sr(t,3))}else{let n=Math.min(e.length,t.count);for(let r=0;r<n;r++){let n=e[r];t.setXYZ(r,n.x,n.y,n.z||0)}e.length>t.count&&W(`BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry.`),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new tr);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){G(`BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.`,this),this.boundingBox.set(new q(-1/0,-1/0,-1/0),new q(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let e=0,n=t.length;e<n;e++){let n=t[e];jr.setFromBufferAttribute(n),this.morphTargetsRelative?(Nr.addVectors(this.boundingBox.min,jr.min),this.boundingBox.expandByPoint(Nr),Nr.addVectors(this.boundingBox.max,jr.max),this.boundingBox.expandByPoint(Nr)):(this.boundingBox.expandByPoint(jr.min),this.boundingBox.expandByPoint(jr.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&G(`BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.`,this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Er);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){G(`BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.`,this),this.boundingSphere.set(new q,1/0);return}if(e){let n=this.boundingSphere.center;if(jr.setFromBufferAttribute(e),t)for(let e=0,n=t.length;e<n;e++){let n=t[e];Mr.setFromBufferAttribute(n),this.morphTargetsRelative?(Nr.addVectors(jr.min,Mr.min),jr.expandByPoint(Nr),Nr.addVectors(jr.max,Mr.max),jr.expandByPoint(Nr)):(jr.expandByPoint(Mr.min),jr.expandByPoint(Mr.max))}jr.getCenter(n);let r=0;for(let t=0,i=e.count;t<i;t++)Nr.fromBufferAttribute(e,t),r=Math.max(r,n.distanceToSquared(Nr));if(t)for(let i=0,a=t.length;i<a;i++){let a=t[i],o=this.morphTargetsRelative;for(let t=0,i=a.count;t<i;t++)Nr.fromBufferAttribute(a,t),o&&(Ar.fromBufferAttribute(e,t),Nr.add(Ar)),r=Math.max(r,n.distanceToSquared(Nr))}this.boundingSphere.radius=Math.sqrt(r),isNaN(this.boundingSphere.radius)&&G(`BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.`,this)}}computeTangents(){let e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){G(`BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)`);return}let n=t.position,r=t.normal,i=t.uv,a=this.getAttribute(`tangent`);(a===void 0||a.count!==n.count)&&(a=new yr(new Float32Array(4*n.count),4),this.setAttribute(`tangent`,a));let o=[],s=[];for(let e=0;e<n.count;e++)o[e]=new q,s[e]=new q;let c=new q,l=new q,u=new q,d=new Mt,f=new Mt,p=new Mt,m=new q,h=new q;function g(e,t,r){c.fromBufferAttribute(n,e),l.fromBufferAttribute(n,t),u.fromBufferAttribute(n,r),d.fromBufferAttribute(i,e),f.fromBufferAttribute(i,t),p.fromBufferAttribute(i,r),l.sub(c),u.sub(c),f.sub(d),p.sub(d);let a=1/(f.x*p.y-p.x*f.y);isFinite(a)&&(m.copy(l).multiplyScalar(p.y).addScaledVector(u,-f.y).multiplyScalar(a),h.copy(u).multiplyScalar(f.x).addScaledVector(l,-p.x).multiplyScalar(a),o[e].add(m),o[t].add(m),o[r].add(m),s[e].add(h),s[t].add(h),s[r].add(h))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let t=0,n=_.length;t<n;++t){let n=_[t],r=n.start,i=n.count;for(let t=r,n=r+i;t<n;t+=3)g(e.getX(t+0),e.getX(t+1),e.getX(t+2))}let v=new q,y=new q,b=new q,x=new q;function S(e){b.fromBufferAttribute(r,e),x.copy(b);let t=o[e];v.copy(t),v.sub(b.multiplyScalar(b.dot(t))).normalize(),y.crossVectors(x,t);let n=y.dot(s[e])<0?-1:1;a.setXYZW(e,v.x,v.y,v.z,n)}for(let t=0,n=_.length;t<n;++t){let n=_[t],r=n.start,i=n.count;for(let t=r,n=r+i;t<n;t+=3)S(e.getX(t+0)),S(e.getX(t+1)),S(e.getX(t+2))}this._transformed=!0}computeVertexNormals(){let e=this.index,t=this.getAttribute(`position`);if(t!==void 0){let n=this.getAttribute(`normal`);if(n===void 0||n.count!==t.count)n=new yr(new Float32Array(t.count*3),3),this.setAttribute(`normal`,n);else for(let e=0,t=n.count;e<t;e++)n.setXYZ(e,0,0,0);let r=new q,i=new q,a=new q,o=new q,s=new q,c=new q,l=new q,u=new q;if(e)for(let d=0,f=e.count;d<f;d+=3){let f=e.getX(d+0),p=e.getX(d+1),m=e.getX(d+2);r.fromBufferAttribute(t,f),i.fromBufferAttribute(t,p),a.fromBufferAttribute(t,m),l.subVectors(a,i),u.subVectors(r,i),l.cross(u),o.fromBufferAttribute(n,f),s.fromBufferAttribute(n,p),c.fromBufferAttribute(n,m),o.add(l),s.add(l),c.add(l),n.setXYZ(f,o.x,o.y,o.z),n.setXYZ(p,s.x,s.y,s.z),n.setXYZ(m,c.x,c.y,c.z)}else for(let e=0,o=t.count;e<o;e+=3)r.fromBufferAttribute(t,e+0),i.fromBufferAttribute(t,e+1),a.fromBufferAttribute(t,e+2),l.subVectors(a,i),u.subVectors(r,i),l.cross(u),n.setXYZ(e+0,l.x,l.y,l.z),n.setXYZ(e+1,l.x,l.y,l.z),n.setXYZ(e+2,l.x,l.y,l.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)Nr.fromBufferAttribute(e,t),Nr.normalize(),e.setXYZ(t,Nr.x,Nr.y,Nr.z)}toNonIndexed(){function t(e,t){let n=e.array,r=e.itemSize,i=e.normalized,a=new n.constructor(t.length*r),o=0,s=0;for(let i=0,c=t.length;i<c;i++){o=e.isInterleavedBufferAttribute?t[i]*e.data.stride+e.offset:t[i]*r;for(let e=0;e<r;e++)a[s++]=n[o++]}return new yr(a,r,i)}if(this.index===null)return W(`BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed.`),this;let n=new e,r=this.index.array,i=this.attributes;for(let e in i){let a=i[e],o=t(a,r);n.setAttribute(e,o)}let a=this.morphAttributes;for(let e in a){let i=[],o=a[e];for(let e=0,n=o.length;e<n;e++){let n=o[e],a=t(n,r);i.push(a)}n.morphAttributes[e]=i}n.morphTargetsRelative=this.morphTargetsRelative;let o=this.groups;for(let e=0,t=o.length;e<t;e++){let t=o[e];n.addGroup(t.start,t.count,t.materialIndex)}return n}toJSON(){let e={metadata:{version:4.7,type:`BufferGeometry`,generator:`BufferGeometry.toJSON`}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?`BufferGeometry`:this.type,this.name!==``&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let t=this.parameters;for(let n in t)t[n]!==void 0&&(e[n]=t[n]);return e}e.data={attributes:{}};let t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});let n=this.attributes;for(let t in n){let r=n[t];e.data.attributes[t]=r.toJSON(e.data)}let r={},i=!1;for(let t in this.morphAttributes){let n=this.morphAttributes[t],a=[];for(let t=0,r=n.length;t<r;t++){let r=n[t];a.push(r.toJSON(e.data))}a.length>0&&(r[t]=a,i=!0)}i&&(e.data.morphAttributes=r,e.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let t={};this.name=e.name;let n=e.index;n!==null&&this.setIndex(n.clone());let r=e.attributes;for(let e in r){let n=r[e];this.setAttribute(e,n.clone(t))}let i=e.morphAttributes;for(let e in i){let n=[],r=i[e];for(let e=0,i=r.length;e<i;e++)n.push(r[e].clone(t));this.morphAttributes[e]=n}this.morphTargetsRelative=e.morphTargetsRelative;let a=e.groups;for(let e=0,t=a.length;e<t;e++){let t=a[e];this.addGroup(t.start,t.count,t.materialIndex)}let o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());let s=e.boundingSphere;return s!==null&&(this.boundingSphere=s.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:`dispose`})}},Fr=class{constructor(e,t){this.isInterleavedBuffer=!0,this.array=e,this.stride=t,this.count=e===void 0?0:e.length/t,this.usage=ut,this.updateRanges=[],this.version=0,this.uuid=Dt()}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.array=new e.array.constructor(e.array),this.count=e.count,this.stride=e.stride,this.usage=e.usage,this}copyAt(e,t,n){e*=this.stride,n*=t.stride;for(let r=0,i=this.stride;r<i;r++)this.array[e+r]=t.array[n+r];return this}set(e,t=0){return this.array.set(e,t),this}clone(e){e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=Dt()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);let t=new this.array.constructor(e.arrayBuffers[this.array.buffer._uuid]),n=new this.constructor(t,this.stride);return n.setUsage(this.usage),n}onUpload(e){return this.onUploadCallback=e,this}toJSON(e){return e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=Dt()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}},Ir=new q,Lr=class e{constructor(e,t,n,r=!1){this.isInterleavedBufferAttribute=!0,this.name=``,this.data=e,this.itemSize=t,this.offset=n,this.normalized=r}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(e){this.data.needsUpdate=e}applyMatrix4(e){for(let t=0,n=this.data.count;t<n;t++)Ir.fromBufferAttribute(this,t),Ir.applyMatrix4(e),this.setXYZ(t,Ir.x,Ir.y,Ir.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)Ir.fromBufferAttribute(this,t),Ir.applyNormalMatrix(e),this.setXYZ(t,Ir.x,Ir.y,Ir.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)Ir.fromBufferAttribute(this,t),Ir.transformDirection(e),this.setXYZ(t,Ir.x,Ir.y,Ir.z);return this}getComponent(e,t){let n=this.array[e*this.data.stride+this.offset+t];return this.normalized&&(n=At(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=jt(n,this.array)),this.data.array[e*this.data.stride+this.offset+t]=n,this}setX(e,t){return this.normalized&&(t=jt(t,this.array)),this.data.array[e*this.data.stride+this.offset]=t,this}setY(e,t){return this.normalized&&(t=jt(t,this.array)),this.data.array[e*this.data.stride+this.offset+1]=t,this}setZ(e,t){return this.normalized&&(t=jt(t,this.array)),this.data.array[e*this.data.stride+this.offset+2]=t,this}setW(e,t){return this.normalized&&(t=jt(t,this.array)),this.data.array[e*this.data.stride+this.offset+3]=t,this}getX(e){let t=this.data.array[e*this.data.stride+this.offset];return this.normalized&&(t=At(t,this.array)),t}getY(e){let t=this.data.array[e*this.data.stride+this.offset+1];return this.normalized&&(t=At(t,this.array)),t}getZ(e){let t=this.data.array[e*this.data.stride+this.offset+2];return this.normalized&&(t=At(t,this.array)),t}getW(e){let t=this.data.array[e*this.data.stride+this.offset+3];return this.normalized&&(t=At(t,this.array)),t}setXY(e,t,n){return e=e*this.data.stride+this.offset,this.normalized&&(t=jt(t,this.array),n=jt(n,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=n,this}setXYZ(e,t,n,r){return e=e*this.data.stride+this.offset,this.normalized&&(t=jt(t,this.array),n=jt(n,this.array),r=jt(r,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=n,this.data.array[e+2]=r,this}setXYZW(e,t,n,r,i){return e=e*this.data.stride+this.offset,this.normalized&&(t=jt(t,this.array),n=jt(n,this.array),r=jt(r,this.array),i=jt(i,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=n,this.data.array[e+2]=r,this.data.array[e+3]=i,this}clone(t){if(t===void 0){vt(`InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.`);let e=[];for(let t=0;t<this.count;t++){let n=t*this.data.stride+this.offset;for(let t=0;t<this.itemSize;t++)e.push(this.data.array[n+t])}return new yr(new this.array.constructor(e),this.itemSize,this.normalized)}else return t.interleavedBuffers===void 0&&(t.interleavedBuffers={}),t.interleavedBuffers[this.data.uuid]===void 0&&(t.interleavedBuffers[this.data.uuid]=this.data.clone(t)),new e(t.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(e){if(e===void 0){vt(`InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.`);let e=[];for(let t=0;t<this.count;t++){let n=t*this.data.stride+this.offset;for(let t=0;t<this.itemSize;t++)e.push(this.data.array[n+t])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:e,normalized:this.normalized}}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.toJSON(e)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}},Rr=0,zr=class extends Ct{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,`id`,{value:Rr++}),this.uuid=Dt(),this.name=``,this.type=`Material`,this.blending=1,this.side=0,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=204,this.blendDst=205,this.blendEquation=100,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Ln(0,0,0),this.blendAlpha=0,this.depthFunc=3,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=519,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=lt,this.stencilZFail=lt,this.stencilZPass=lt,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let t in e){let n=e[t];if(n===void 0){W(`Material: parameter '${t}' has value of undefined.`);continue}let r=this[t];if(r===void 0){W(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}r&&r.isColor?r.set(n):r&&r.isVector2&&n&&n.isVector2||r&&r.isEuler&&n&&n.isEuler||r&&r.isVector3&&n&&n.isVector3?r.copy(n):this[t]=n}}toJSON(e){let t=e===void 0||typeof e==`string`;t&&(e={textures:{},images:{}});let n={metadata:{version:4.7,type:`Material`,generator:`Material.toJSON`}};n.uuid=this.uuid,n.type=this.type,this.name!==``&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==1&&(n.blending=this.blending),this.side!==0&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==204&&(n.blendSrc=this.blendSrc),this.blendDst!==205&&(n.blendDst=this.blendDst),this.blendEquation!==100&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==3&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==519&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==7680&&(n.stencilFail=this.stencilFail),this.stencilZFail!==7680&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==7680&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.allowOverride===!1&&(n.allowOverride=!1),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!==`round`&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!==`round`&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function r(e){let t=[];for(let n in e){let r=e[n];delete r.metadata,t.push(r)}return t}if(t){let t=r(e.textures),i=r(e.images);t.length>0&&(n.textures=t),i.length>0&&(n.images=i)}return n}fromJSON(e,t){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new Ln().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessRange=e.iridescenceThicknessRange),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors==`number`?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=t[e.map]||null),e.matcap!==void 0&&(this.matcap=t[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=t[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=t[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=t[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let t=e.normalScale;Array.isArray(t)===!1&&(t=[t,t]),this.normalScale=new Mt().fromArray(t)}return e.displacementMap!==void 0&&(this.displacementMap=t[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=t[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=t[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=t[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=t[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=t[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=t[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=t[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=t[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=t[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=t[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=t[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=t[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=t[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new Mt().fromArray(e.clearcoatNormalScale)),e.iridescenceMap!==void 0&&(this.iridescenceMap=t[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=t[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=t[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=t[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=t[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=t[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=t[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let t=e.clippingPlanes,n=null;if(t!==null){let e=t.length;n=Array(e);for(let r=0;r!==e;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:`dispose`})}set needsUpdate(e){e===!0&&this.version++}},Br=class extends zr{constructor(e){super(),this.isSpriteMaterial=!0,this.type=`SpriteMaterial`,this.color=new Ln(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.rotation=e.rotation,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},Vr,Hr=new q,Ur=new q,Wr=new q,Gr=new Mt,Kr=new Mt,qr=new tn,Jr=new q,Yr=new q,Xr=new q,Zr=new Mt,Qr=new Mt,$r=new Mt,ei=class extends kn{constructor(e=new Br){if(super(),this.isSprite=!0,this.type=`Sprite`,Vr===void 0){Vr=new Pr;let e=new Fr(new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),5);Vr.setIndex([0,1,2,0,2,3]),Vr.setAttribute(`position`,new Lr(e,3,0,!1)),Vr.setAttribute(`uv`,new Lr(e,2,3,!1))}this.geometry=Vr,this.material=e,this.center=new Mt(.5,.5),this.count=1}raycast(e,t){e.camera===null&&G(`Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.`),Ur.setFromMatrixScale(this.matrixWorld),qr.copy(e.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(e.camera.matrixWorldInverse,this.matrixWorld),Wr.setFromMatrixPosition(this.modelViewMatrix),e.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&Ur.multiplyScalar(-Wr.z);let n=this.material.rotation,r,i;n!==0&&(i=Math.cos(n),r=Math.sin(n));let a=this.center;ti(Jr.set(-.5,-.5,0),Wr,a,Ur,r,i),ti(Yr.set(.5,-.5,0),Wr,a,Ur,r,i),ti(Xr.set(.5,.5,0),Wr,a,Ur,r,i),Zr.set(0,0),Qr.set(1,0),$r.set(1,1);let o=e.ray.intersectTriangle(Jr,Yr,Xr,!1,Hr);if(o===null&&(ti(Yr.set(-.5,.5,0),Wr,a,Ur,r,i),Qr.set(0,1),o=e.ray.intersectTriangle(Jr,Xr,Yr,!1,Hr),o===null))return;let s=e.ray.origin.distanceTo(Hr);s<e.near||s>e.far||t.push({distance:s,point:Hr.clone(),uv:er.getInterpolation(Hr,Jr,Yr,Xr,Zr,Qr,$r,new Mt),face:null,object:this})}copy(e,t){return super.copy(e,t),e.center!==void 0&&this.center.copy(e.center),this.material=e.material,this}};function ti(e,t,n,r,i,a){Gr.subVectors(e,n).addScalar(.5).multiply(r),i===void 0?Kr.copy(Gr):(Kr.x=a*Gr.x-i*Gr.y,Kr.y=i*Gr.x+a*Gr.y),e.copy(t),e.x+=Kr.x,e.y+=Kr.y,e.applyMatrix4(qr)}var ni=new q,ri=new q,ii=new q,ai=new q,oi=new q,si=new q,ci=new q,li=class{constructor(e=new q,t=new q(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,ni)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);let n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let t=ni.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(ni.copy(this.origin).addScaledVector(this.direction,t),ni.distanceToSquared(e))}distanceSqToSegment(e,t,n,r){ri.copy(e).add(t).multiplyScalar(.5),ii.copy(t).sub(e).normalize(),ai.copy(this.origin).sub(ri);let i=e.distanceTo(t)*.5,a=-this.direction.dot(ii),o=ai.dot(this.direction),s=-ai.dot(ii),c=ai.lengthSq(),l=Math.abs(1-a*a),u,d,f,p;if(l>0)if(u=a*s-o,d=a*o-s,p=i*l,u>=0)if(d>=-p)if(d<=p){let e=1/l;u*=e,d*=e,f=u*(u+a*d+2*o)+d*(a*u+d+2*s)+c}else d=i,u=Math.max(0,-(a*d+o)),f=-u*u+d*(d+2*s)+c;else d=-i,u=Math.max(0,-(a*d+o)),f=-u*u+d*(d+2*s)+c;else d<=-p?(u=Math.max(0,-(-a*i+o)),d=u>0?-i:Math.min(Math.max(-i,-s),i),f=-u*u+d*(d+2*s)+c):d<=p?(u=0,d=Math.min(Math.max(-i,-s),i),f=d*(d+2*s)+c):(u=Math.max(0,-(a*i+o)),d=u>0?i:Math.min(Math.max(-i,-s),i),f=-u*u+d*(d+2*s)+c);else d=a>0?-i:i,u=Math.max(0,-(a*d+o)),f=-u*u+d*(d+2*s)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,u),r&&r.copy(ri).addScaledVector(ii,d),f}intersectSphere(e,t){ni.subVectors(e.center,this.origin);let n=ni.dot(this.direction),r=ni.dot(ni)-n*n,i=e.radius*e.radius;if(r>i)return null;let a=Math.sqrt(i-r),o=n-a,s=n+a;return s<0?null:o<0?this.at(s,t):this.at(o,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;let n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){let n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){let t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,r,i,a,o,s,c=1/this.direction.x,l=1/this.direction.y,u=1/this.direction.z,d=this.origin;return c>=0?(n=(e.min.x-d.x)*c,r=(e.max.x-d.x)*c):(n=(e.max.x-d.x)*c,r=(e.min.x-d.x)*c),l>=0?(i=(e.min.y-d.y)*l,a=(e.max.y-d.y)*l):(i=(e.max.y-d.y)*l,a=(e.min.y-d.y)*l),n>a||i>r||((i>n||isNaN(n))&&(n=i),(a<r||isNaN(r))&&(r=a),u>=0?(o=(e.min.z-d.z)*u,s=(e.max.z-d.z)*u):(o=(e.max.z-d.z)*u,s=(e.min.z-d.z)*u),n>s||o>r)||((o>n||n!==n)&&(n=o),(s<r||r!==r)&&(r=s),r<0)?null:this.at(n>=0?n:r,t)}intersectsBox(e){return this.intersectBox(e,ni)!==null}intersectTriangle(e,t,n,r,i){oi.subVectors(t,e),si.subVectors(n,e),ci.crossVectors(oi,si);let a=this.direction.dot(ci),o;if(a>0){if(r)return null;o=1}else if(a<0)o=-1,a=-a;else return null;ai.subVectors(this.origin,e);let s=o*this.direction.dot(si.crossVectors(ai,si));if(s<0)return null;let c=o*this.direction.dot(oi.cross(ai));if(c<0||s+c>a)return null;let l=-o*ai.dot(ci);return l<0?null:this.at(l/a,i)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},ui=class extends zr{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type=`MeshBasicMaterial`,this.color=new Ln(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new fn,this.combine=0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap=`round`,this.wireframeLinejoin=`round`,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},di=new tn,fi=new li,pi=new Er,mi=new q,hi=new q,gi=new q,_i=new q,vi=new q,yi=new q,bi=new q,xi=new q,Si=class extends kn{constructor(e=new Pr,t=new ui){super(),this.isMesh=!0,this.type=`Mesh`,this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let e=this.geometry.morphAttributes,t=Object.keys(e);if(t.length>0){let n=e[t[0]];if(n!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let e=0,t=n.length;e<t;e++){let t=n[e].name||String(e);this.morphTargetInfluences.push(0),this.morphTargetDictionary[t]=e}}}}getVertexPosition(e,t){let n=this.geometry,r=n.attributes.position,i=n.morphAttributes.position,a=n.morphTargetsRelative;t.fromBufferAttribute(r,e);let o=this.morphTargetInfluences;if(i&&o){yi.set(0,0,0);for(let n=0,r=i.length;n<r;n++){let r=o[n],s=i[n];r!==0&&(vi.fromBufferAttribute(s,e),a?yi.addScaledVector(vi,r):yi.addScaledVector(vi.sub(t),r))}t.add(yi)}return t}raycast(e,t){let n=this.geometry,r=this.material,i=this.matrixWorld;r!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),pi.copy(n.boundingSphere),pi.applyMatrix4(i),fi.copy(e.ray).recast(e.near),!(pi.containsPoint(fi.origin)===!1&&(fi.intersectSphere(pi,mi)===null||fi.origin.distanceToSquared(mi)>(e.far-e.near)**2))&&(di.copy(i).invert(),fi.copy(e.ray).applyMatrix4(di),!(n.boundingBox!==null&&fi.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,fi)))}_computeIntersections(e,t,n){let r,i=this.geometry,a=this.material,o=i.index,s=i.attributes.position,c=i.attributes.uv,l=i.attributes.uv1,u=i.attributes.normal,d=i.groups,f=i.drawRange;if(o!==null)if(Array.isArray(a))for(let i=0,s=d.length;i<s;i++){let s=d[i],p=a[s.materialIndex],m=Math.max(s.start,f.start),h=Math.min(o.count,Math.min(s.start+s.count,f.start+f.count));for(let i=m,a=h;i<a;i+=3){let a=o.getX(i),d=o.getX(i+1),f=o.getX(i+2);r=wi(this,p,e,n,c,l,u,a,d,f),r&&(r.faceIndex=Math.floor(i/3),r.face.materialIndex=s.materialIndex,t.push(r))}}else{let i=Math.max(0,f.start),s=Math.min(o.count,f.start+f.count);for(let d=i,f=s;d<f;d+=3){let i=o.getX(d),s=o.getX(d+1),f=o.getX(d+2);r=wi(this,a,e,n,c,l,u,i,s,f),r&&(r.faceIndex=Math.floor(d/3),t.push(r))}}else if(s!==void 0)if(Array.isArray(a))for(let i=0,o=d.length;i<o;i++){let o=d[i],p=a[o.materialIndex],m=Math.max(o.start,f.start),h=Math.min(s.count,Math.min(o.start+o.count,f.start+f.count));for(let i=m,a=h;i<a;i+=3){let a=i,s=i+1,d=i+2;r=wi(this,p,e,n,c,l,u,a,s,d),r&&(r.faceIndex=Math.floor(i/3),r.face.materialIndex=o.materialIndex,t.push(r))}}else{let i=Math.max(0,f.start),o=Math.min(s.count,f.start+f.count);for(let s=i,d=o;s<d;s+=3){let i=s,o=s+1,d=s+2;r=wi(this,a,e,n,c,l,u,i,o,d),r&&(r.faceIndex=Math.floor(s/3),t.push(r))}}}};function Ci(e,t,n,r,i,a,o,s){let c;if(c=t.side===1?r.intersectTriangle(o,a,i,!0,s):r.intersectTriangle(i,a,o,t.side===0,s),c===null)return null;xi.copy(s),xi.applyMatrix4(e.matrixWorld);let l=n.ray.origin.distanceTo(xi);return l<n.near||l>n.far?null:{distance:l,point:xi.clone(),object:e}}function wi(e,t,n,r,i,a,o,s,c,l){e.getVertexPosition(s,hi),e.getVertexPosition(c,gi),e.getVertexPosition(l,_i);let u=Ci(e,t,n,r,hi,gi,_i,bi);if(u){let e=new q;er.getBarycoord(bi,hi,gi,_i,e),i&&(u.uv=er.getInterpolatedAttribute(i,s,c,l,e,new Mt)),a&&(u.uv1=er.getInterpolatedAttribute(a,s,c,l,e,new Mt)),o&&(u.normal=er.getInterpolatedAttribute(o,s,c,l,e,new q),u.normal.dot(r.direction)>0&&u.normal.multiplyScalar(-1));let t={a:s,b:c,c:l,normal:new q,materialIndex:0};er.getNormal(hi,gi,_i,t.normal),u.face=t,u.barycoord=e}return u}var Ti=class extends Yt{constructor(e=null,t=1,n=1,r,i,a,o,s,c=A,l=A,u,d){super(null,a,o,s,c,l,r,i,u,d),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}},Ei=new q,Di=new q,Oi=new J,ki=class{constructor(e=new q(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,r){return this.normal.set(e,t,n),this.constant=r,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){let r=Ei.subVectors(n,t).cross(Di.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(r,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t,n=!0){let r=e.delta(Ei),i=this.normal.dot(r);if(i===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;let a=-(e.start.dot(this.normal)+this.constant)/i;return n===!0&&(a<0||a>1)?null:t.copy(e.start).addScaledVector(r,a)}intersectsLine(e){let t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){let n=t||Oi.getNormalMatrix(e),r=this.coplanarPoint(Ei).applyMatrix4(e),i=this.normal.applyMatrix3(n).normalize();return this.constant=-r.dot(i),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}},Ai=new Er,ji=new Mt(.5,.5),Mi=new q,Ni=class{constructor(e=new ki,t=new ki,n=new ki,r=new ki,i=new ki,a=new ki){this.planes=[e,t,n,r,i,a]}set(e,t,n,r,i,a){let o=this.planes;return o[0].copy(e),o[1].copy(t),o[2].copy(n),o[3].copy(r),o[4].copy(i),o[5].copy(a),this}copy(e){let t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=dt,n=!1){let r=this.planes,i=e.elements,a=i[0],o=i[1],s=i[2],c=i[3],l=i[4],u=i[5],d=i[6],f=i[7],p=i[8],m=i[9],h=i[10],g=i[11],_=i[12],v=i[13],y=i[14],b=i[15];if(r[0].setComponents(c-a,f-l,g-p,b-_).normalize(),r[1].setComponents(c+a,f+l,g+p,b+_).normalize(),r[2].setComponents(c+o,f+u,g+m,b+v).normalize(),r[3].setComponents(c-o,f-u,g-m,b-v).normalize(),n)r[4].setComponents(s,d,h,y).normalize(),r[5].setComponents(c-s,f-d,g-h,b-y).normalize();else if(r[4].setComponents(c-s,f-d,g-h,b-y).normalize(),t===2e3)r[5].setComponents(c+s,f+d,g+h,b+y).normalize();else if(t===2001)r[5].setComponents(s,d,h,y).normalize();else throw Error(`THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: `+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Ai.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Ai.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Ai)}intersectsSprite(e){return Ai.center.set(0,0,0),Ai.radius=.7071067811865476+ji.distanceTo(e.center),Ai.applyMatrix4(e.matrixWorld),this.intersectsSphere(Ai)}intersectsSphere(e){let t=this.planes,n=e.center,r=-e.radius;for(let e=0;e<6;e++)if(t[e].distanceToPoint(n)<r)return!1;return!0}intersectsBox(e){let t=this.planes;for(let n=0;n<6;n++){let r=t[n];if(Mi.x=r.normal.x>0?e.max.x:e.min.x,Mi.y=r.normal.y>0?e.max.y:e.min.y,Mi.z=r.normal.z>0?e.max.z:e.min.z,r.distanceToPoint(Mi)<0)return!1}return!0}containsPoint(e){let t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}},Pi=class extends zr{constructor(e){super(),this.isPointsMaterial=!0,this.type=`PointsMaterial`,this.color=new Ln(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},Fi=new tn,Ii=new li,Li=new Er,Ri=new q,zi=class extends kn{constructor(e=new Pr,t=new Pi){super(),this.isPoints=!0,this.type=`Points`,this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){let n=this.geometry,r=this.matrixWorld,i=e.params.Points.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Li.copy(n.boundingSphere),Li.applyMatrix4(r),Li.radius+=i,e.ray.intersectsSphere(Li)===!1)return;Fi.copy(r).invert(),Ii.copy(e.ray).applyMatrix4(Fi);let o=i/((this.scale.x+this.scale.y+this.scale.z)/3),s=o*o,c=n.index,l=n.attributes.position;if(c!==null){let n=Math.max(0,a.start),i=Math.min(c.count,a.start+a.count);for(let a=n,o=i;a<o;a++){let n=c.getX(a);Ri.fromBufferAttribute(l,n),Bi(Ri,n,s,r,e,t,this)}}else{let n=Math.max(0,a.start),i=Math.min(l.count,a.start+a.count);for(let a=n,o=i;a<o;a++)Ri.fromBufferAttribute(l,a),Bi(Ri,a,s,r,e,t,this)}}updateMorphTargets(){let e=this.geometry.morphAttributes,t=Object.keys(e);if(t.length>0){let n=e[t[0]];if(n!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let e=0,t=n.length;e<t;e++){let t=n[e].name||String(e);this.morphTargetInfluences.push(0),this.morphTargetDictionary[t]=e}}}}};function Bi(e,t,n,r,i,a,o){let s=Ii.distanceSqToPoint(e);if(s<n){let n=new q;Ii.closestPointToPoint(e,n),n.applyMatrix4(r);let c=i.ray.origin.distanceTo(n);if(c<i.near||c>i.far)return;a.push({distance:c,distanceToRay:Math.sqrt(s),point:n,index:t,face:null,faceIndex:null,barycoord:null,object:o})}}var Vi=class extends Yt{constructor(e=[],t=301,n,r,i,a,o,s,c,l){super(e,t,n,r,i,a,o,s,c,l),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}},Hi=class extends Yt{constructor(e,t,n,r,i,a,o,s,c){super(e,t,n,r,i,a,o,s,c),this.isCanvasTexture=!0,this.needsUpdate=!0}},Ui=class extends Yt{constructor(e,t,n=ne,r,i,a,o=A,s=A,c,l=V,u=1){if(l!==1026&&l!==1027)throw Error(`THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat`);super({width:e,height:t,depth:u},r,i,a,o,s,l,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Gt(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}},Wi=class extends Ui{constructor(e,t=ne,n=301,r,i,a=A,o=A,s,c=V){let l={width:e,height:e,depth:1},u=[l,l,l,l,l,l];super(e,e,t,n,r,i,a,o,s,c),this.image=u,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},Gi=class extends Yt{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},Ki=class e extends Pr{constructor(e=1,t=1,n=1,r=1,i=1,a=1){super(),this.type=`BoxGeometry`,this.parameters={width:e,height:t,depth:n,widthSegments:r,heightSegments:i,depthSegments:a};let o=this;r=Math.floor(r),i=Math.floor(i),a=Math.floor(a);let s=[],c=[],l=[],u=[],d=0,f=0;p(`z`,`y`,`x`,-1,-1,n,t,e,a,i,0),p(`z`,`y`,`x`,1,-1,n,t,-e,a,i,1),p(`x`,`z`,`y`,1,1,e,n,t,r,a,2),p(`x`,`z`,`y`,1,-1,e,n,-t,r,a,3),p(`x`,`y`,`z`,1,-1,e,t,n,r,i,4),p(`x`,`y`,`z`,-1,-1,e,t,-n,r,i,5),this.setIndex(s),this.setAttribute(`position`,new Sr(c,3)),this.setAttribute(`normal`,new Sr(l,3)),this.setAttribute(`uv`,new Sr(u,2));function p(e,t,n,r,i,a,p,m,h,g,_){let v=a/h,y=p/g,b=a/2,x=p/2,S=m/2,C=h+1,w=g+1,T=0,E=0,D=new q;for(let a=0;a<w;a++){let o=a*y-x;for(let s=0;s<C;s++)D[e]=(s*v-b)*r,D[t]=o*i,D[n]=S,c.push(D.x,D.y,D.z),D[e]=0,D[t]=0,D[n]=m>0?1:-1,l.push(D.x,D.y,D.z),u.push(s/h),u.push(1-a/g),T+=1}for(let e=0;e<g;e++)for(let t=0;t<h;t++){let n=d+t+C*e,r=d+t+C*(e+1),i=d+(t+1)+C*(e+1),a=d+(t+1)+C*e;s.push(n,r,a),s.push(r,i,a),E+=6}o.addGroup(f,E,_),f+=E,d+=T}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(t){return new e(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}},qi=class e extends Pr{constructor(e=1,t=1,n=1,r=1){super(),this.type=`PlaneGeometry`,this.parameters={width:e,height:t,widthSegments:n,heightSegments:r};let i=e/2,a=t/2,o=Math.floor(n),s=Math.floor(r),c=o+1,l=s+1,u=e/o,d=t/s,f=[],p=[],m=[],h=[];for(let e=0;e<l;e++){let t=e*d-a;for(let n=0;n<c;n++){let r=n*u-i;p.push(r,-t,0),m.push(0,0,1),h.push(n/o),h.push(1-e/s)}}for(let e=0;e<s;e++)for(let t=0;t<o;t++){let n=t+c*e,r=t+c*(e+1),i=t+1+c*(e+1),a=t+1+c*e;f.push(n,r,a),f.push(r,i,a)}this.setIndex(f),this.setAttribute(`position`,new Sr(p,3)),this.setAttribute(`normal`,new Sr(m,3)),this.setAttribute(`uv`,new Sr(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(t){return new e(t.width,t.height,t.widthSegments,t.heightSegments)}};function Ji(e){let t={};for(let n in e){t[n]={};for(let r in e[n]){let i=e[n][r];if(Xi(i))i.isRenderTargetTexture?(W(`UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms().`),t[n][r]=null):t[n][r]=i.clone();else if(Array.isArray(i))if(Xi(i[0])){let e=[];for(let t=0,n=i.length;t<n;t++)e[t]=i[t].clone();t[n][r]=e}else t[n][r]=i.slice();else t[n][r]=i}}return t}function Yi(e){let t={};for(let n=0;n<e.length;n++){let r=Ji(e[n]);for(let e in r)t[e]=r[e]}return t}function Xi(e){return e&&(e.isColor||e.isMatrix3||e.isMatrix4||e.isVector2||e.isVector3||e.isVector4||e.isTexture||e.isQuaternion)}function Zi(e){let t=[];for(let n=0;n<e.length;n++)t.push(e[n].clone());return t}function Qi(e){let t=e.getRenderTarget();return t===null?e.outputColorSpace:t.isXRRenderTarget===!0?t.texture.colorSpace:Y.workingColorSpace}var $i={clone:Ji,merge:Yi},ea=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,ta=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,na=class extends zr{constructor(e){super(),this.isShaderMaterial=!0,this.type=`ShaderMaterial`,this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=ea,this.fragmentShader=ta,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Ji(e.uniforms),this.uniformsGroups=Zi(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(let n in this.uniforms){let r=this.uniforms[n].value;r&&r.isTexture?t.uniforms[n]={type:`t`,value:r.toJSON(e).uuid}:r&&r.isColor?t.uniforms[n]={type:`c`,value:r.getHex()}:r&&r.isVector2?t.uniforms[n]={type:`v2`,value:r.toArray()}:r&&r.isVector3?t.uniforms[n]={type:`v3`,value:r.toArray()}:r&&r.isVector4?t.uniforms[n]={type:`v4`,value:r.toArray()}:r&&r.isMatrix3?t.uniforms[n]={type:`m3`,value:r.toArray()}:r&&r.isMatrix4?t.uniforms[n]={type:`m4`,value:r.toArray()}:t.uniforms[n]={value:r}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;let n={};for(let e in this.extensions)this.extensions[e]===!0&&(n[e]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}fromJSON(e,t){if(super.fromJSON(e,t),e.uniforms!==void 0)for(let n in e.uniforms){let r=e.uniforms[n];switch(this.uniforms[n]={},r.type){case`t`:this.uniforms[n].value=t[r.value]||null;break;case`c`:this.uniforms[n].value=new Ln().setHex(r.value);break;case`v2`:this.uniforms[n].value=new Mt().fromArray(r.value);break;case`v3`:this.uniforms[n].value=new q().fromArray(r.value);break;case`v4`:this.uniforms[n].value=new Xt().fromArray(r.value);break;case`m3`:this.uniforms[n].value=new J().fromArray(r.value);break;case`m4`:this.uniforms[n].value=new tn().fromArray(r.value);break;default:this.uniforms[n].value=r.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let t in e.extensions)this.extensions[t]=e.extensions[t];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},ra=class extends na{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type=`RawShaderMaterial`}},ia=class extends zr{constructor(e){super(),this.isMeshLambertMaterial=!0,this.type=`MeshLambertMaterial`,this.color=new Ln(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Ln(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=0,this.normalScale=new Mt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new fn,this.combine=0,this.reflectivity=1,this.envMapIntensity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap=`round`,this.wireframeLinejoin=`round`,this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.envMapIntensity=e.envMapIntensity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}},aa=class extends zr{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type=`MeshDepthMaterial`,this.depthPacking=it,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},oa=class extends zr{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type=`MeshDistanceMaterial`,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function sa(e,t){return!e||e.constructor===t?e:typeof t.BYTES_PER_ELEMENT==`number`?new t(e):Array.prototype.slice.call(e)}var ca=class{constructor(e,t,n,r){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=r===void 0?new t.constructor(n):r,this.sampleValues=t,this.valueSize=n,this.settings=null,this.DefaultSettings_={}}evaluate(e){let t=this.parameterPositions,n=this._cachedIndex,r=t[n],i=t[n-1];validate_interval:{seek:{let a;linear_scan:{forward_scan:if(!(e<r)){for(let a=n+2;;){if(r===void 0){if(e<i)break forward_scan;return n=t.length,this._cachedIndex=n,this.copySampleValue_(n-1)}if(n===a)break;if(i=r,r=t[++n],e<r)break seek}a=t.length;break linear_scan}if(!(e>=i)){let o=t[1];e<o&&(n=2,i=o);for(let a=n-2;;){if(i===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(n===a)break;if(r=i,i=t[--n-1],e>=i)break seek}a=n,n=0;break linear_scan}break validate_interval}for(;n<a;){let r=n+a>>>1;e<t[r]?a=r:n=r+1}if(r=t[n],i=t[n-1],i===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(r===void 0)return n=t.length,this._cachedIndex=n,this.copySampleValue_(n-1)}this._cachedIndex=n,this.intervalChanged_(n,i,r)}return this.interpolate_(n,i,e,r)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let t=this.resultBuffer,n=this.sampleValues,r=this.valueSize,i=e*r;for(let e=0;e!==r;++e)t[e]=n[i+e];return t}interpolate_(){throw Error(`THREE.Interpolant: Call to abstract method.`)}intervalChanged_(){}},la=class extends ca{constructor(e,t,n,r){super(e,t,n,r),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:tt,endingEnd:tt}}intervalChanged_(e,t,n){let r=this.parameterPositions,i=e-2,a=e+1,o=r[i],s=r[a];if(o===void 0)switch(this.getSettings_().endingStart){case nt:i=e,o=2*t-n;break;case rt:i=r.length-2,o=t+r[i]-r[i+1];break;default:i=e,o=n}if(s===void 0)switch(this.getSettings_().endingEnd){case nt:a=e,s=2*n-t;break;case rt:a=1,s=n+r[1]-r[0];break;default:a=e-1,s=t}let c=(n-t)*.5,l=this.valueSize;this._weightPrev=c/(t-o),this._weightNext=c/(s-n),this._offsetPrev=i*l,this._offsetNext=a*l}interpolate_(e,t,n,r){let i=this.resultBuffer,a=this.sampleValues,o=this.valueSize,s=e*o,c=s-o,l=this._offsetPrev,u=this._offsetNext,d=this._weightPrev,f=this._weightNext,p=(n-t)/(r-t),m=p*p,h=m*p,g=-d*h+2*d*m-d*p,_=(1+d)*h+(-1.5-2*d)*m+(-.5+d)*p+1,v=(-1-f)*h+(1.5+f)*m+.5*p,y=f*h-f*m;for(let e=0;e!==o;++e)i[e]=g*a[l+e]+_*a[c+e]+v*a[s+e]+y*a[u+e];return i}},ua=class extends ca{constructor(e,t,n,r){super(e,t,n,r)}interpolate_(e,t,n,r){let i=this.resultBuffer,a=this.sampleValues,o=this.valueSize,s=e*o,c=s-o,l=(n-t)/(r-t),u=1-l;for(let e=0;e!==o;++e)i[e]=a[c+e]*u+a[s+e]*l;return i}},da=class extends ca{constructor(e,t,n,r){super(e,t,n,r)}interpolate_(e){return this.copySampleValue_(e-1)}},fa=class extends ca{interpolate_(e,t,n,r){let i=this.resultBuffer,a=this.sampleValues,o=this.valueSize,s=e*o,c=s-o,l=this.inTangents,u=this.outTangents;if(!l||!u){let e=(n-t)/(r-t),l=1-e;for(let t=0;t!==o;++t)i[t]=a[c+t]*l+a[s+t]*e;return i}let d=o*2,f=e-1;for(let p=0;p!==o;++p){let o=a[c+p],m=a[s+p],h=f*d+p*2,g=u[h],_=u[h+1],v=e*d+p*2,y=l[v],b=l[v+1],x=(n-t)/(r-t),S,C,w,T,E;for(let e=0;e<8;e++){S=x*x,C=S*x,w=1-x,T=w*w,E=T*w;let e=E*t+3*T*x*g+3*w*S*y+C*r-n;if(Math.abs(e)<1e-10)break;let i=3*T*(g-t)+6*w*x*(y-g)+3*S*(r-y);if(Math.abs(i)<1e-10)break;x-=e/i,x=Math.max(0,Math.min(1,x))}i[p]=E*o+3*T*x*_+3*w*S*b+C*m}return i}},pa=class{constructor(e,t,n,r){if(e===void 0)throw Error(`THREE.KeyframeTrack: track name is undefined`);if(t===void 0||t.length===0)throw Error(`THREE.KeyframeTrack: no keyframes in track named `+e);this.name=e,this.times=sa(t,this.TimeBufferType),this.values=sa(n,this.ValueBufferType),this.setInterpolation(r||this.DefaultInterpolation)}static toJSON(e){let t=e.constructor,n;if(t.toJSON!==this.toJSON)n=t.toJSON(e);else{n={name:e.name,times:sa(e.times,Array),values:sa(e.values,Array)};let t=e.getInterpolation();t!==e.DefaultInterpolation&&(n.interpolation=t)}return n.type=e.ValueTypeName,n}InterpolantFactoryMethodDiscrete(e){return new da(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new ua(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new la(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let t=new fa(this.times,this.values,this.getValueSize(),e);return this.settings&&(t.inTangents=this.settings.inTangents,t.outTangents=this.settings.outTangents),t}setInterpolation(e){let t;switch(e){case Ze:t=this.InterpolantFactoryMethodDiscrete;break;case Qe:t=this.InterpolantFactoryMethodLinear;break;case $e:t=this.InterpolantFactoryMethodSmooth;break;case et:t=this.InterpolantFactoryMethodBezier;break}if(t===void 0){let t=`unsupported interpolation for `+this.ValueTypeName+` keyframe track named `+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw Error(t);return W(`KeyframeTrack:`,t),this}return this.createInterpolant=t,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return Ze;case this.InterpolantFactoryMethodLinear:return Qe;case this.InterpolantFactoryMethodSmooth:return $e;case this.InterpolantFactoryMethodBezier:return et}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let t=this.times;for(let n=0,r=t.length;n!==r;++n)t[n]+=e}return this}scale(e){if(e!==1){let t=this.times;for(let n=0,r=t.length;n!==r;++n)t[n]*=e}return this}trim(e,t){let n=this.times,r=n.length,i=0,a=r-1;for(;i!==r&&n[i]<e;)++i;for(;a!==-1&&n[a]>t;)--a;if(++a,i!==0||a!==r){i>=a&&(a=Math.max(a,1),i=a-1);let e=this.getValueSize();this.times=n.slice(i,a),this.values=this.values.slice(i*e,a*e)}return this}validate(){let e=!0,t=this.getValueSize();t-Math.floor(t)!==0&&(G(`KeyframeTrack: Invalid value size in track.`,this),e=!1);let n=this.times,r=this.values,i=n.length;i===0&&(G(`KeyframeTrack: Track is empty.`,this),e=!1);let a=null;for(let t=0;t!==i;t++){let r=n[t];if(typeof r==`number`&&isNaN(r)){G(`KeyframeTrack: Time is not a valid number.`,this,t,r),e=!1;break}if(a!==null&&a>r){G(`KeyframeTrack: Out of order keys.`,this,t,r,a),e=!1;break}a=r}if(r!==void 0&&pt(r))for(let t=0,n=r.length;t!==n;++t){let n=r[t];if(isNaN(n)){G(`KeyframeTrack: Value is not a valid number.`,this,t,n),e=!1;break}}return e}optimize(){let e=this.times.slice(),t=this.values.slice(),n=this.getValueSize(),r=this.getInterpolation()===$e,i=e.length-1,a=1;for(let o=1;o<i;++o){let i=!1,s=e[o];if(s!==e[o+1]&&(o!==1||s!==e[0]))if(r)i=!0;else{let e=o*n,r=e-n,a=e+n;for(let o=0;o!==n;++o){let n=t[e+o];if(n!==t[r+o]||n!==t[a+o]){i=!0;break}}}if(i){if(o!==a){e[a]=e[o];let r=o*n,i=a*n;for(let e=0;e!==n;++e)t[i+e]=t[r+e]}++a}}if(i>0){e[a]=e[i];for(let e=i*n,r=a*n,o=0;o!==n;++o)t[r+o]=t[e+o];++a}return a===e.length?(this.times=e,this.values=t):(this.times=e.slice(0,a),this.values=t.slice(0,a*n)),this}clone(){let e=this.times.slice(),t=this.values.slice(),n=this.constructor,r=new n(this.name,e,t);return r.createInterpolant=this.createInterpolant,r}};pa.prototype.ValueTypeName=``,pa.prototype.TimeBufferType=Float32Array,pa.prototype.ValueBufferType=Float32Array,pa.prototype.DefaultInterpolation=Qe;var ma=class extends pa{constructor(e,t,n){super(e,t,n)}};ma.prototype.ValueTypeName=`bool`,ma.prototype.ValueBufferType=Array,ma.prototype.DefaultInterpolation=Ze,ma.prototype.InterpolantFactoryMethodLinear=void 0,ma.prototype.InterpolantFactoryMethodSmooth=void 0;var ha=class extends pa{constructor(e,t,n,r){super(e,t,n,r)}};ha.prototype.ValueTypeName=`color`;var ga=class extends pa{constructor(e,t,n,r){super(e,t,n,r)}};ga.prototype.ValueTypeName=`number`;var _a=class extends ca{constructor(e,t,n,r){super(e,t,n,r)}interpolate_(e,t,n,r){let i=this.resultBuffer,a=this.sampleValues,o=this.valueSize,s=(n-t)/(r-t),c=e*o;for(let e=c+o;c!==e;c+=4)Nt.slerpFlat(i,0,a,c-o,a,c,s);return i}},va=class extends pa{constructor(e,t,n,r){super(e,t,n,r)}InterpolantFactoryMethodLinear(e){return new _a(this.times,this.values,this.getValueSize(),e)}};va.prototype.ValueTypeName=`quaternion`,va.prototype.InterpolantFactoryMethodSmooth=void 0;var ya=class extends pa{constructor(e,t,n){super(e,t,n)}};ya.prototype.ValueTypeName=`string`,ya.prototype.ValueBufferType=Array,ya.prototype.DefaultInterpolation=Ze,ya.prototype.InterpolantFactoryMethodLinear=void 0,ya.prototype.InterpolantFactoryMethodSmooth=void 0;var ba=class extends pa{constructor(e,t,n,r){super(e,t,n,r)}};ba.prototype.ValueTypeName=`vector`;var xa=new class{constructor(e,t,n){let r=this,i=!1,a=0,o=0,s,c=[];this.onStart=void 0,this.onLoad=e,this.onProgress=t,this.onError=n,this._abortController=null,this.itemStart=function(e){o++,i===!1&&r.onStart!==void 0&&r.onStart(e,a,o),i=!0},this.itemEnd=function(e){a++,r.onProgress!==void 0&&r.onProgress(e,a,o),a===o&&(i=!1,r.onLoad!==void 0&&r.onLoad())},this.itemError=function(e){r.onError!==void 0&&r.onError(e)},this.resolveURL=function(e){return e=e.normalize(`NFC`),s?s(e):e},this.setURLModifier=function(e){return s=e,this},this.addHandler=function(e,t){return c.push(e,t),this},this.removeHandler=function(e){let t=c.indexOf(e);return t!==-1&&c.splice(t,2),this},this.getHandler=function(e){for(let t=0,n=c.length;t<n;t+=2){let n=c[t],r=c[t+1];if(n.global&&(n.lastIndex=0),n.test(e))return r}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||=new AbortController,this._abortController}},Sa=class{constructor(e){this.manager=e===void 0?xa:e,this.crossOrigin=`anonymous`,this.withCredentials=!1,this.path=``,this.resourcePath=``,this.requestHeader={},typeof __THREE_DEVTOOLS__<`u`&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent(`observe`,{detail:this}))}load(){}loadAsync(e,t){let n=this;return new Promise(function(r,i){n.load(e,r,t,i)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}};Sa.DEFAULT_MATERIAL_NAME=`__DEFAULT`;var Ca=class extends kn{constructor(e,t=1){super(),this.isLight=!0,this.type=`Light`,this.color=new Ln(e),this.intensity=t}dispose(){this.dispatchEvent({type:`dispose`})}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){let t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,t}},wa=class extends Ca{constructor(e,t,n){super(e,n),this.isHemisphereLight=!0,this.type=`HemisphereLight`,this.position.copy(kn.DEFAULT_UP),this.updateMatrix(),this.groundColor=new Ln(t)}copy(e,t){return super.copy(e,t),this.groundColor.copy(e.groundColor),this}toJSON(e){let t=super.toJSON(e);return t.object.groundColor=this.groundColor.getHex(),t}},Ta=new tn,Ea=new q,Da=new q,Oa=class{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new Mt(512,512),this.mapType=I,this.map=null,this.mapPass=null,this.matrix=new tn,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new Ni,this._frameExtents=new Mt(1,1),this._viewportCount=1,this._viewports=[new Xt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){let t=this.camera,n=this.matrix;Ea.setFromMatrixPosition(e.matrixWorld),t.position.copy(Ea),Da.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(Da),t.updateMatrixWorld(),Ta.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(Ta,t.coordinateSystem,t.reversedDepth),t.coordinateSystem===2001||t.reversedDepth?n.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(Ta)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this.biasNode=e.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){let e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}},ka=new q,Aa=new Nt,ja=new q,Ma=class extends kn{constructor(){super(),this.isCamera=!0,this.type=`Camera`,this.matrixWorldInverse=new tn,this.projectionMatrix=new tn,this.projectionMatrixInverse=new tn,this.coordinateSystem=dt,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(ka,Aa,ja),ja.x===1&&ja.y===1&&ja.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(ka,Aa,ja.set(1,1,1)).invert()}updateWorldMatrix(e,t,n=!1){super.updateWorldMatrix(e,t,n),this.matrixWorld.decompose(ka,Aa,ja),ja.x===1&&ja.y===1&&ja.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(ka,Aa,ja.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},Na=new q,Pa=new Mt,Fa=new Mt,Ia=class extends Ma{constructor(e=50,t=1,n=.1,r=2e3){super(),this.isPerspectiveCamera=!0,this.type=`PerspectiveCamera`,this.fov=e,this.zoom=1,this.near=n,this.far=r,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let t=.5*this.getFilmHeight()/e;this.fov=Et*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(Tt*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return Et*2*Math.atan(Math.tan(Tt*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){Na.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Na.x,Na.y).multiplyScalar(-e/Na.z),Na.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Na.x,Na.y).multiplyScalar(-e/Na.z)}getViewSize(e,t){return this.getViewBounds(e,Pa,Fa),t.subVectors(Fa,Pa)}setViewOffset(e,t,n,r,i,a){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=r,this.view.width=i,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,t=e*Math.tan(Tt*.5*this.fov)/this.zoom,n=2*t,r=this.aspect*n,i=-.5*r,a=this.view;if(this.view!==null&&this.view.enabled){let e=a.fullWidth,o=a.fullHeight;i+=a.offsetX*r/e,t-=a.offsetY*n/o,r*=a.width/e,n*=a.height/o}let o=this.filmOffset;o!==0&&(i+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(i,i+r,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}},La=class extends Ma{constructor(e=-1,t=1,n=1,r=-1,i=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type=`OrthographicCamera`,this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=r,this.near=i,this.far=a,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,r,i,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=r,this.view.width=i,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,r=(this.top+this.bottom)/2,i=n-e,a=n+e,o=r+t,s=r-t;if(this.view!==null&&this.view.enabled){let e=(this.right-this.left)/this.view.fullWidth/this.zoom,t=(this.top-this.bottom)/this.view.fullHeight/this.zoom;i+=e*this.view.offsetX,a=i+e*this.view.width,o-=t*this.view.offsetY,s=o-t*this.view.height}this.projectionMatrix.makeOrthographic(i,a,o,s,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}},Ra=class extends Oa{constructor(){super(new La(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}},za=class extends Ca{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type=`DirectionalLight`,this.position.copy(kn.DEFAULT_UP),this.updateMatrix(),this.target=new kn,this.shadow=new Ra}dispose(){super.dispose(),this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.shadow=this.shadow.toJSON(),t.object.target=this.target.uuid,t}},Ba=-90,Va=1,Ha=class extends kn{constructor(e,t,n){super(),this.type=`CubeCamera`,this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;let r=new Ia(Ba,Va,e,t);r.layers=this.layers,this.add(r);let i=new Ia(Ba,Va,e,t);i.layers=this.layers,this.add(i);let a=new Ia(Ba,Va,e,t);a.layers=this.layers,this.add(a);let o=new Ia(Ba,Va,e,t);o.layers=this.layers,this.add(o);let s=new Ia(Ba,Va,e,t);s.layers=this.layers,this.add(s);let c=new Ia(Ba,Va,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){let e=this.coordinateSystem,t=this.children.concat(),[n,r,i,a,o,s]=t;for(let e of t)this.remove(e);if(e===2e3)n.up.set(0,1,0),n.lookAt(1,0,0),r.up.set(0,1,0),r.lookAt(-1,0,0),i.up.set(0,0,-1),i.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),s.up.set(0,1,0),s.lookAt(0,0,-1);else if(e===2001)n.up.set(0,-1,0),n.lookAt(-1,0,0),r.up.set(0,-1,0),r.lookAt(1,0,0),i.up.set(0,0,1),i.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),s.up.set(0,-1,0),s.lookAt(0,0,-1);else throw Error(`THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: `+e);for(let e of t)this.add(e),e.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();let{renderTarget:n,activeMipmapLevel:r}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[i,a,o,s,c,l]=this.children,u=e.getRenderTarget(),d=e.getActiveCubeFace(),f=e.getActiveMipmapLevel(),p=e.xr.enabled;e.xr.enabled=!1;let m=n.texture.generateMipmaps;n.texture.generateMipmaps=!1;let h=!1;h=e.isWebGLRenderer===!0?e.state.buffers.depth.getReversed():e.reversedDepthBuffer,e.setRenderTarget(n,0,r),h&&e.autoClear===!1&&e.clearDepth(),e.render(t,i),e.setRenderTarget(n,1,r),h&&e.autoClear===!1&&e.clearDepth(),e.render(t,a),e.setRenderTarget(n,2,r),h&&e.autoClear===!1&&e.clearDepth(),e.render(t,o),e.setRenderTarget(n,3,r),h&&e.autoClear===!1&&e.clearDepth(),e.render(t,s),e.setRenderTarget(n,4,r),h&&e.autoClear===!1&&e.clearDepth(),e.render(t,c),n.texture.generateMipmaps=m,e.setRenderTarget(n,5,r),h&&e.autoClear===!1&&e.clearDepth(),e.render(t,l),e.setRenderTarget(u,d,f),e.xr.enabled=p,n.texture.needsPMREMUpdate=!0}},Ua=class extends Ia{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}},Wa=`\\[\\]\\.:\\/`,Ga=RegExp(`[`+Wa+`]`,`g`),Ka=`[^`+Wa+`]`,qa=`[^`+Wa.replace(`\\.`,``)+`]`,Ja=`((?:WC+[\\/:])*)`.replace(`WC`,Ka),Ya=`(WCOD+)?`.replace(`WCOD`,qa),Xa=`(?:\\.(WC+)(?:\\[(.+)\\])?)?`.replace(`WC`,Ka),Za=`\\.(WC+)(?:\\[(.+)\\])?`.replace(`WC`,Ka),Qa=RegExp(`^`+Ja+Ya+Xa+Za+`$`),$a=[`material`,`materials`,`bones`,`map`],eo=class{constructor(e,t,n){let r=n||to.parseTrackName(t);this._targetGroup=e,this._bindings=e.subscribe_(t,r)}getValue(e,t){this.bind();let n=this._targetGroup.nCachedObjects_,r=this._bindings[n];r!==void 0&&r.getValue(e,t)}setValue(e,t){let n=this._bindings;for(let r=this._targetGroup.nCachedObjects_,i=n.length;r!==i;++r)n[r].setValue(e,t)}bind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,n=e.length;t!==n;++t)e[t].bind()}unbind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,n=e.length;t!==n;++t)e[t].unbind()}},to=class e{constructor(t,n,r){this.path=n,this.parsedPath=r||e.parseTrackName(n),this.node=e.findNode(t,this.parsedPath.nodeName),this.rootNode=t,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(t,n,r){return t&&t.isAnimationObjectGroup?new e.Composite(t,n,r):new e(t,n,r)}static sanitizeNodeName(e){return e.replace(/\s/g,`_`).replace(Ga,``)}static parseTrackName(e){let t=Qa.exec(e);if(t===null)throw Error(`THREE.PropertyBinding: Cannot parse trackName: `+e);let n={nodeName:t[2],objectName:t[3],objectIndex:t[4],propertyName:t[5],propertyIndex:t[6]},r=n.nodeName&&n.nodeName.lastIndexOf(`.`);if(r!==void 0&&r!==-1){let e=n.nodeName.substring(r+1);$a.indexOf(e)!==-1&&(n.nodeName=n.nodeName.substring(0,r),n.objectName=e)}if(n.propertyName===null||n.propertyName.length===0)throw Error(`THREE.PropertyBinding: can not parse propertyName from trackName: `+e);return n}static findNode(e,t){if(t===void 0||t===``||t===`.`||t===-1||t===e.name||t===e.uuid)return e;if(e.skeleton){let n=e.skeleton.getBoneByName(t);if(n!==void 0)return n}if(e.children){let n=function(e){for(let r=0;r<e.length;r++){let i=e[r];if(i.name===t||i.uuid===t)return i;let a=n(i.children);if(a)return a}return null},r=n(e.children);if(r)return r}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,t){e[t]=this.targetObject[this.propertyName]}_getValue_array(e,t){let n=this.resolvedProperty;for(let r=0,i=n.length;r!==i;++r)e[t++]=n[r]}_getValue_arrayElement(e,t){e[t]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,t){this.resolvedProperty.toArray(e,t)}_setValue_direct(e,t){this.targetObject[this.propertyName]=e[t]}_setValue_direct_setNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,t){let n=this.resolvedProperty;for(let r=0,i=n.length;r!==i;++r)n[r]=e[t++]}_setValue_array_setNeedsUpdate(e,t){let n=this.resolvedProperty;for(let r=0,i=n.length;r!==i;++r)n[r]=e[t++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,t){let n=this.resolvedProperty;for(let r=0,i=n.length;r!==i;++r)n[r]=e[t++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,t){this.resolvedProperty[this.propertyIndex]=e[t]}_setValue_arrayElement_setNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,t){this.resolvedProperty.fromArray(e,t)}_setValue_fromArray_setNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,t){this.bind(),this.getValue(e,t)}_setValue_unbound(e,t){this.bind(),this.setValue(e,t)}bind(){let t=this.node,n=this.parsedPath,r=n.objectName,i=n.propertyName,a=n.propertyIndex;if(t||(t=e.findNode(this.rootNode,n.nodeName),this.node=t),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!t){W(`PropertyBinding: No target node found for track: `+this.path+`.`);return}if(r){let e=n.objectIndex;switch(r){case`materials`:if(!t.material){G(`PropertyBinding: Can not bind to material as node does not have a material.`,this);return}if(!t.material.materials){G(`PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.`,this);return}t=t.material.materials;break;case`bones`:if(!t.skeleton){G(`PropertyBinding: Can not bind to bones as node does not have a skeleton.`,this);return}t=t.skeleton.bones;for(let n=0;n<t.length;n++)if(t[n].name===e){e=n;break}break;case`map`:if(`map`in t){t=t.map;break}if(!t.material){G(`PropertyBinding: Can not bind to material as node does not have a material.`,this);return}if(!t.material.map){G(`PropertyBinding: Can not bind to material.map as node.material does not have a map.`,this);return}t=t.material.map;break;default:if(t[r]===void 0){G(`PropertyBinding: Can not bind to objectName of node undefined.`,this);return}t=t[r]}if(e!==void 0){if(t[e]===void 0){G(`PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.`,this,t);return}t=t[e]}}let o=t[i];if(o===void 0){let e=n.nodeName;G(`PropertyBinding: Trying to update property for track: `+e+`.`+i+` but it wasn't found.`,t);return}let s=this.Versioning.None;this.targetObject=t,t.isMaterial===!0?s=this.Versioning.NeedsUpdate:t.isObject3D===!0&&(s=this.Versioning.MatrixWorldNeedsUpdate);let c=this.BindingType.Direct;if(a!==void 0){if(i===`morphTargetInfluences`){if(!t.geometry){G(`PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.`,this);return}if(!t.geometry.morphAttributes){G(`PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.`,this);return}t.morphTargetDictionary[a]!==void 0&&(a=t.morphTargetDictionary[a])}c=this.BindingType.ArrayElement,this.resolvedProperty=o,this.propertyIndex=a}else o.fromArray!==void 0&&o.toArray!==void 0?(c=this.BindingType.HasFromToArray,this.resolvedProperty=o):Array.isArray(o)?(c=this.BindingType.EntireArray,this.resolvedProperty=o):this.propertyName=i;this.getValue=this.GetterByBindingType[c],this.setValue=this.SetterByBindingTypeAndVersioning[c][s]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};to.Composite=eo,to.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3},to.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2},to.prototype.GetterByBindingType=[to.prototype._getValue_direct,to.prototype._getValue_array,to.prototype._getValue_arrayElement,to.prototype._getValue_toArray],to.prototype.SetterByBindingTypeAndVersioning=[[to.prototype._setValue_direct,to.prototype._setValue_direct_setNeedsUpdate,to.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[to.prototype._setValue_array,to.prototype._setValue_array_setNeedsUpdate,to.prototype._setValue_array_setMatrixWorldNeedsUpdate],[to.prototype._setValue_arrayElement,to.prototype._setValue_arrayElement_setNeedsUpdate,to.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[to.prototype._setValue_fromArray,to.prototype._setValue_fromArray_setNeedsUpdate,to.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]],class e{static{e.prototype.isMatrix2=!0}constructor(e,t,n,r){this.elements=[1,0,0,1],e!==void 0&&this.set(e,t,n,r)}identity(){return this.set(1,0,0,1),this}fromArray(e,t=0){for(let n=0;n<4;n++)this.elements[n]=e[n+t];return this}set(e,t,n,r){let i=this.elements;return i[0]=e,i[2]=t,i[1]=n,i[3]=r,this}};function no(e,t,n,r){let i=ro(r);switch(n){case le:return e*t;case fe:return e*t/i.components*i.byteLength;case pe:return e*t/i.components*i.byteLength;case me:return e*t*2/i.components*i.byteLength;case he:return e*t*2/i.components*i.byteLength;case B:return e*t*3/i.components*i.byteLength;case ue:return e*t*4/i.components*i.byteLength;case ge:return e*t*4/i.components*i.byteLength;case _e:case ve:return Math.floor((e+3)/4)*Math.floor((t+3)/4)*8;case ye:case be:return Math.floor((e+3)/4)*Math.floor((t+3)/4)*16;case Se:case we:return Math.max(e,16)*Math.max(t,8)/4;case xe:case Ce:return Math.max(e,8)*Math.max(t,8)/2;case Te:case Ee:case De:case Oe:return Math.floor((e+3)/4)*Math.floor((t+3)/4)*8;case H:case ke:case U:return Math.floor((e+3)/4)*Math.floor((t+3)/4)*16;case Ae:return Math.floor((e+3)/4)*Math.floor((t+3)/4)*16;case je:return Math.floor((e+4)/5)*Math.floor((t+3)/4)*16;case Me:return Math.floor((e+4)/5)*Math.floor((t+4)/5)*16;case Ne:return Math.floor((e+5)/6)*Math.floor((t+4)/5)*16;case Pe:return Math.floor((e+5)/6)*Math.floor((t+5)/6)*16;case Fe:return Math.floor((e+7)/8)*Math.floor((t+4)/5)*16;case Ie:return Math.floor((e+7)/8)*Math.floor((t+5)/6)*16;case Le:return Math.floor((e+7)/8)*Math.floor((t+7)/8)*16;case Re:return Math.floor((e+9)/10)*Math.floor((t+4)/5)*16;case ze:return Math.floor((e+9)/10)*Math.floor((t+5)/6)*16;case Be:return Math.floor((e+9)/10)*Math.floor((t+7)/8)*16;case Ve:return Math.floor((e+9)/10)*Math.floor((t+9)/10)*16;case He:return Math.floor((e+11)/12)*Math.floor((t+9)/10)*16;case Ue:return Math.floor((e+11)/12)*Math.floor((t+11)/12)*16;case We:case Ge:case Ke:return Math.ceil(e/4)*Math.ceil(t/4)*16;case qe:case Je:return Math.ceil(e/4)*Math.ceil(t/4)*8;case Ye:case Xe:return Math.ceil(e/4)*Math.ceil(t/4)*16}throw Error(`Unable to determine texture byte length for ${n} format.`)}function ro(e){switch(e){case I:case ee:return{byteLength:1,components:1};case L:case te:case ie:return{byteLength:2,components:1};case z:case ae:return{byteLength:2,components:4};case ne:case R:case re:return{byteLength:4,components:1};case se:case ce:return{byteLength:4,components:3}}throw Error(`THREE.TextureUtils: Unknown texture type ${e}.`)}typeof __THREE_DEVTOOLS__<`u`&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent(`register`,{detail:{revision:`185`}})),typeof window<`u`&&(window.__THREE__?W(`WARNING: Multiple instances of Three.js being imported.`):window.__THREE__=`185`);function io(){let e=null,t=!1,n=null,r=null;function i(t,a){n(t,a),r=e.requestAnimationFrame(i)}return{start:function(){t!==!0&&n!==null&&e!==null&&(r=e.requestAnimationFrame(i),t=!0)},stop:function(){e!==null&&e.cancelAnimationFrame(r),t=!1},setAnimationLoop:function(e){n=e},setContext:function(t){e=t}}}function ao(e){let t=new WeakMap;function n(t,n){let r=t.array,i=t.usage,a=r.byteLength,o=e.createBuffer();e.bindBuffer(n,o),e.bufferData(n,r,i),t.onUploadCallback();let s;if(r instanceof Float32Array)s=e.FLOAT;else if(typeof Float16Array<`u`&&r instanceof Float16Array)s=e.HALF_FLOAT;else if(r instanceof Uint16Array)s=t.isFloat16BufferAttribute?e.HALF_FLOAT:e.UNSIGNED_SHORT;else if(r instanceof Int16Array)s=e.SHORT;else if(r instanceof Uint32Array)s=e.UNSIGNED_INT;else if(r instanceof Int32Array)s=e.INT;else if(r instanceof Int8Array)s=e.BYTE;else if(r instanceof Uint8Array)s=e.UNSIGNED_BYTE;else if(r instanceof Uint8ClampedArray)s=e.UNSIGNED_BYTE;else throw Error(`THREE.WebGLAttributes: Unsupported buffer data format: `+r);return{buffer:o,type:s,bytesPerElement:r.BYTES_PER_ELEMENT,version:t.version,size:a}}function r(t,n,r){let i=n.array,a=n.updateRanges;if(e.bindBuffer(r,t),a.length===0)e.bufferSubData(r,0,i);else{a.sort((e,t)=>e.start-t.start);let t=0;for(let e=1;e<a.length;e++){let n=a[t],r=a[e];r.start<=n.start+n.count+1?n.count=Math.max(n.count,r.start+r.count-n.start):(++t,a[t]=r)}a.length=t+1;for(let t=0,n=a.length;t<n;t++){let n=a[t];e.bufferSubData(r,n.start*i.BYTES_PER_ELEMENT,i,n.start,n.count)}n.clearUpdateRanges()}n.onUploadCallback()}function i(e){return e.isInterleavedBufferAttribute&&(e=e.data),t.get(e)}function a(n){n.isInterleavedBufferAttribute&&(n=n.data);let r=t.get(n);r&&(e.deleteBuffer(r.buffer),t.delete(n))}function o(e,i){if(e.isInterleavedBufferAttribute&&(e=e.data),e.isGLBufferAttribute){let n=t.get(e);(!n||n.version<e.version)&&t.set(e,{buffer:e.buffer,type:e.type,bytesPerElement:e.elementSize,version:e.version});return}let a=t.get(e);if(a===void 0)t.set(e,n(e,i));else if(a.version<e.version){if(a.size!==e.array.byteLength)throw Error(`THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.`);r(a.buffer,e,i),a.version=e.version}}return{get:i,remove:a,update:o}}var oo={alphahash_fragment:`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,alphahash_pars_fragment:`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,alphamap_fragment:`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,alphamap_pars_fragment:`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,alphatest_fragment:`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,alphatest_pars_fragment:`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,aomap_fragment:`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,aomap_pars_fragment:`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,batching_pars_vertex:`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,batching_vertex:`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,begin_vertex:`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,beginnormal_vertex:`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,bsdfs:`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,iridescence_fragment:`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,bumpmap_pars_fragment:`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,clipping_planes_fragment:`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,clipping_planes_pars_fragment:`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,clipping_planes_pars_vertex:`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,clipping_planes_vertex:`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,color_fragment:`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,color_pars_fragment:`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,color_pars_vertex:`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,color_vertex:`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,common:`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
#define inverseTransformDirection transformDirectionByInverseViewMatrix
vec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {
	return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );
}
vec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {
	return normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,cube_uv_reflection_fragment:`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,defaultnormal_vertex:`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
#endif`,displacementmap_pars_vertex:`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,displacementmap_vertex:`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,emissivemap_fragment:`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,emissivemap_pars_fragment:`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,colorspace_fragment:`gl_FragColor = linearToOutputTexel( gl_FragColor );`,colorspace_pars_fragment:`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,envmap_fragment:`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,envmap_common_pars_fragment:`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,envmap_pars_fragment:`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,envmap_pars_vertex:`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,envmap_physical_pars_fragment:`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,envmap_vertex:`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,fog_vertex:`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,fog_pars_vertex:`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,fog_fragment:`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,fog_pars_fragment:`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,gradientmap_pars_fragment:`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,lightmap_pars_fragment:`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,lights_lambert_fragment:`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,lights_lambert_pars_fragment:`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,lights_pars_begin:`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif
#include <lightprobes_pars_fragment>`,lights_toon_fragment:`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,lights_toon_pars_fragment:`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,lights_phong_fragment:`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,lights_phong_pars_fragment:`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,lights_physical_fragment:`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,lights_physical_pars_fragment:`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		return 0.5 / max( gv + gl, EPSILON );
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,lights_fragment_begin:`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
	#ifdef USE_LIGHT_PROBES_GRID
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		irradiance += getLightProbeGridIrradiance( probeWorldPos, probeWorldNormal );
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,lights_fragment_maps:`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,lights_fragment_end:`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,lightprobes_pars_fragment:`#ifdef USE_LIGHT_PROBES_GRID
uniform highp sampler3D probesSH;
uniform vec3 probesMin;
uniform vec3 probesMax;
uniform vec3 probesResolution;
vec3 getLightProbeGridIrradiance( vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 resMinusOne = res - 1.0;
	vec3 probeSpacing = gridRange / resMinusOne;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.0, 1.0 );
	uvw = uvw * resMinusOne / res + 0.5 / res;
	float nz          = res.z;
	float paddedSlices = nz + 2.0;
	float atlasDepth  = 7.0 * paddedSlices;
	float uvZBase     = uvw.z * nz + 1.0;
	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );
	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );
	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );
	vec3 c0 = s0.xyz;
	vec3 c1 = vec3( s0.w, s1.xy );
	vec3 c2 = vec3( s1.zw, s2.x );
	vec3 c3 = s2.yzw;
	vec3 c4 = s3.xyz;
	vec3 c5 = vec3( s3.w, s4.xy );
	vec3 c6 = vec3( s4.zw, s5.x );
	vec3 c7 = s5.yzw;
	vec3 c8 = s6.xyz;
	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;
	vec3 result = c0 * 0.886227;
	result += c1 * 2.0 * 0.511664 * y;
	result += c2 * 2.0 * 0.511664 * z;
	result += c3 * 2.0 * 0.511664 * x;
	result += c4 * 2.0 * 0.429043 * x * y;
	result += c5 * 2.0 * 0.429043 * y * z;
	result += c6 * ( 0.743125 * z * z - 0.247708 );
	result += c7 * 2.0 * 0.429043 * x * z;
	result += c8 * 0.429043 * ( x * x - y * y );
	return max( result, vec3( 0.0 ) );
}
#endif`,logdepthbuf_fragment:`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,logdepthbuf_pars_fragment:`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,logdepthbuf_pars_vertex:`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,logdepthbuf_vertex:`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,map_fragment:`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,map_pars_fragment:`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,map_particle_fragment:`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,map_particle_pars_fragment:`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,metalnessmap_fragment:`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,metalnessmap_pars_fragment:`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,morphinstance_vertex:`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,morphcolor_vertex:`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,morphnormal_vertex:`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,morphtarget_pars_vertex:`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,morphtarget_vertex:`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,normal_fragment_begin:`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#ifdef DOUBLE_SIDED
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#ifdef DOUBLE_SIDED
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,normal_fragment_maps:`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#if defined( USE_PACKED_NORMALMAP )
		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
	#endif
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,normal_pars_fragment:`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,normal_pars_vertex:`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,normal_vertex:`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,normalmap_pars_fragment:`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,clearcoat_normal_fragment_begin:`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,clearcoat_normal_fragment_maps:`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,clearcoat_pars_fragment:`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,iridescence_pars_fragment:`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,opaque_fragment:`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,packing:`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,premultiplied_alpha_fragment:`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,project_vertex:`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,dithering_fragment:`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,dithering_pars_fragment:`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,roughnessmap_fragment:`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,roughnessmap_pars_fragment:`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,shadowmap_pars_fragment:`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,shadowmap_pars_vertex:`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,shadowmap_vertex:`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,shadowmask_pars_fragment:`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,skinbase_vertex:`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,skinning_pars_vertex:`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,skinning_vertex:`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,skinnormal_vertex:`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,specularmap_fragment:`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,specularmap_pars_fragment:`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,tonemapping_fragment:`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,tonemapping_pars_fragment:`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,transmission_fragment:`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,transmission_pars_fragment:`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,uv_pars_fragment:`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,uv_pars_vertex:`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,uv_vertex:`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,worldpos_vertex:`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,background_vert:`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,background_frag:`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,backgroundCube_vert:`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,backgroundCube_frag:`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,cube_vert:`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,cube_frag:`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,depth_vert:`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,depth_frag:`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,distance_vert:`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,distance_frag:`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,equirect_vert:`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,equirect_frag:`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,linedashed_vert:`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,linedashed_frag:`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,meshbasic_vert:`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,meshbasic_frag:`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,meshlambert_vert:`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,meshlambert_frag:`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,meshmatcap_vert:`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,meshmatcap_frag:`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,meshnormal_vert:`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,meshnormal_frag:`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,meshphong_vert:`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,meshphong_frag:`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,meshphysical_vert:`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,meshphysical_frag:`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,meshtoon_vert:`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,meshtoon_frag:`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,points_vert:`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,points_frag:`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,shadow_vert:`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,shadow_frag:`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,sprite_vert:`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,sprite_frag:`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`},X={common:{diffuse:{value:new Ln(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new J},alphaMap:{value:null},alphaMapTransform:{value:new J},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new J}},envmap:{envMap:{value:null},envMapRotation:{value:new J},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new J}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new J}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new J},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new J},normalScale:{value:new Mt(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new J},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new J}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new J}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new J}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Ln(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new q},probesMax:{value:new q},probesResolution:{value:new q}},points:{diffuse:{value:new Ln(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new J},alphaTest:{value:0},uvTransform:{value:new J}},sprite:{diffuse:{value:new Ln(16777215)},opacity:{value:1},center:{value:new Mt(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new J},alphaMap:{value:null},alphaMapTransform:{value:new J},alphaTest:{value:0}}},so={basic:{uniforms:Yi([X.common,X.specularmap,X.envmap,X.aomap,X.lightmap,X.fog]),vertexShader:oo.meshbasic_vert,fragmentShader:oo.meshbasic_frag},lambert:{uniforms:Yi([X.common,X.specularmap,X.envmap,X.aomap,X.lightmap,X.emissivemap,X.bumpmap,X.normalmap,X.displacementmap,X.fog,X.lights,{emissive:{value:new Ln(0)},envMapIntensity:{value:1}}]),vertexShader:oo.meshlambert_vert,fragmentShader:oo.meshlambert_frag},phong:{uniforms:Yi([X.common,X.specularmap,X.envmap,X.aomap,X.lightmap,X.emissivemap,X.bumpmap,X.normalmap,X.displacementmap,X.fog,X.lights,{emissive:{value:new Ln(0)},specular:{value:new Ln(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:oo.meshphong_vert,fragmentShader:oo.meshphong_frag},standard:{uniforms:Yi([X.common,X.envmap,X.aomap,X.lightmap,X.emissivemap,X.bumpmap,X.normalmap,X.displacementmap,X.roughnessmap,X.metalnessmap,X.fog,X.lights,{emissive:{value:new Ln(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:oo.meshphysical_vert,fragmentShader:oo.meshphysical_frag},toon:{uniforms:Yi([X.common,X.aomap,X.lightmap,X.emissivemap,X.bumpmap,X.normalmap,X.displacementmap,X.gradientmap,X.fog,X.lights,{emissive:{value:new Ln(0)}}]),vertexShader:oo.meshtoon_vert,fragmentShader:oo.meshtoon_frag},matcap:{uniforms:Yi([X.common,X.bumpmap,X.normalmap,X.displacementmap,X.fog,{matcap:{value:null}}]),vertexShader:oo.meshmatcap_vert,fragmentShader:oo.meshmatcap_frag},points:{uniforms:Yi([X.points,X.fog]),vertexShader:oo.points_vert,fragmentShader:oo.points_frag},dashed:{uniforms:Yi([X.common,X.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:oo.linedashed_vert,fragmentShader:oo.linedashed_frag},depth:{uniforms:Yi([X.common,X.displacementmap]),vertexShader:oo.depth_vert,fragmentShader:oo.depth_frag},normal:{uniforms:Yi([X.common,X.bumpmap,X.normalmap,X.displacementmap,{opacity:{value:1}}]),vertexShader:oo.meshnormal_vert,fragmentShader:oo.meshnormal_frag},sprite:{uniforms:Yi([X.sprite,X.fog]),vertexShader:oo.sprite_vert,fragmentShader:oo.sprite_frag},background:{uniforms:{uvTransform:{value:new J},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:oo.background_vert,fragmentShader:oo.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new J}},vertexShader:oo.backgroundCube_vert,fragmentShader:oo.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:oo.cube_vert,fragmentShader:oo.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:oo.equirect_vert,fragmentShader:oo.equirect_frag},distance:{uniforms:Yi([X.common,X.displacementmap,{referencePosition:{value:new q},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:oo.distance_vert,fragmentShader:oo.distance_frag},shadow:{uniforms:Yi([X.lights,X.fog,{color:{value:new Ln(0)},opacity:{value:1}}]),vertexShader:oo.shadow_vert,fragmentShader:oo.shadow_frag}};so.physical={uniforms:Yi([so.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new J},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new J},clearcoatNormalScale:{value:new Mt(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new J},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new J},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new J},sheen:{value:0},sheenColor:{value:new Ln(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new J},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new J},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new J},transmissionSamplerSize:{value:new Mt},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new J},attenuationDistance:{value:0},attenuationColor:{value:new Ln(0)},specularColor:{value:new Ln(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new J},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new J},anisotropyVector:{value:new Mt},anisotropyMap:{value:null},anisotropyMapTransform:{value:new J}}]),vertexShader:oo.meshphysical_vert,fragmentShader:oo.meshphysical_frag};var co={r:0,b:0,g:0},lo=new tn,uo=new J;uo.set(-1,0,0,0,1,0,0,0,1);function fo(e,t,n,r,i,a){let o=new Ln(0),s=i===!0?0:1,c,l,u=null,d=0,f=null;function p(e){let n=e.isScene===!0?e.background:null;if(n&&n.isTexture){let r=e.backgroundBlurriness>0;n=t.get(n,r)}return n}function m(t){let r=!1,i=p(t);i===null?g(o,s):i&&i.isColor&&(g(i,1),r=!0);let c=e.xr.getEnvironmentBlendMode();c===`additive`?n.buffers.color.setClear(0,0,0,1,a):c===`alpha-blend`&&n.buffers.color.setClear(0,0,0,0,a),(e.autoClear||r)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil))}function h(t,n){let i=p(n);i&&(i.isCubeTexture||i.mapping===306)?(l===void 0&&(l=new Si(new Ki(1,1,1),new na({name:`BackgroundCubeMaterial`,uniforms:Ji(so.backgroundCube.uniforms),vertexShader:so.backgroundCube.vertexShader,fragmentShader:so.backgroundCube.fragmentShader,side:1,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute(`normal`),l.geometry.deleteAttribute(`uv`),l.onBeforeRender=function(e,t,n){this.matrixWorld.copyPosition(n.matrixWorld)},Object.defineProperty(l.material,`envMap`,{get:function(){return this.uniforms.envMap.value}}),r.update(l)),l.material.uniforms.envMap.value=i,l.material.uniforms.backgroundBlurriness.value=n.backgroundBlurriness,l.material.uniforms.backgroundIntensity.value=n.backgroundIntensity,l.material.uniforms.backgroundRotation.value.setFromMatrix4(lo.makeRotationFromEuler(n.backgroundRotation)).transpose(),i.isCubeTexture&&i.isRenderTargetTexture===!1&&l.material.uniforms.backgroundRotation.value.premultiply(uo),l.material.toneMapped=Y.getTransfer(i.colorSpace)!==ct,(u!==i||d!==i.version||f!==e.toneMapping)&&(l.material.needsUpdate=!0,u=i,d=i.version,f=e.toneMapping),l.layers.enableAll(),t.unshift(l,l.geometry,l.material,0,0,null)):i&&i.isTexture&&(c===void 0&&(c=new Si(new qi(2,2),new na({name:`BackgroundMaterial`,uniforms:Ji(so.background.uniforms),vertexShader:so.background.vertexShader,fragmentShader:so.background.fragmentShader,side:0,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute(`normal`),Object.defineProperty(c.material,`map`,{get:function(){return this.uniforms.t2D.value}}),r.update(c)),c.material.uniforms.t2D.value=i,c.material.uniforms.backgroundIntensity.value=n.backgroundIntensity,c.material.toneMapped=Y.getTransfer(i.colorSpace)!==ct,i.matrixAutoUpdate===!0&&i.updateMatrix(),c.material.uniforms.uvTransform.value.copy(i.matrix),(u!==i||d!==i.version||f!==e.toneMapping)&&(c.material.needsUpdate=!0,u=i,d=i.version,f=e.toneMapping),c.layers.enableAll(),t.unshift(c,c.geometry,c.material,0,0,null))}function g(t,r){t.getRGB(co,Qi(e)),n.buffers.color.setClear(co.r,co.g,co.b,r,a)}function _(){l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return o},setClearColor:function(e,t=1){o.set(e),s=t,g(o,s)},getClearAlpha:function(){return s},setClearAlpha:function(e){s=e,g(o,s)},render:m,addToRenderList:h,dispose:_}}function po(e,t){let n=e.getParameter(e.MAX_VERTEX_ATTRIBS),r={},i=f(null),a=i,o=!1;function s(n,r,i,s,c){let u=!1,f=d(n,s,i,r);a!==f&&(a=f,l(a.object)),u=p(n,s,i,c),u&&m(n,s,i,c),c!==null&&t.update(c,e.ELEMENT_ARRAY_BUFFER),(u||o)&&(o=!1,b(n,r,i,s),c!==null&&e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,t.get(c).buffer))}function c(){return e.createVertexArray()}function l(t){return e.bindVertexArray(t)}function u(t){return e.deleteVertexArray(t)}function d(e,t,n,i){let a=i.wireframe===!0,o=r[t.id];o===void 0&&(o={},r[t.id]=o);let s=e.isInstancedMesh===!0?e.id:0,l=o[s];l===void 0&&(l={},o[s]=l);let u=l[n.id];u===void 0&&(u={},l[n.id]=u);let d=u[a];return d===void 0&&(d=f(c()),u[a]=d),d}function f(e){let t=[],r=[],i=[];for(let e=0;e<n;e++)t[e]=0,r[e]=0,i[e]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:t,enabledAttributes:r,attributeDivisors:i,object:e,attributes:{},index:null}}function p(e,t,n,r){let i=a.attributes,o=t.attributes,s=0,c=n.getAttributes();for(let t in c)if(c[t].location>=0){let n=i[t],r=o[t];if(r===void 0&&(t===`instanceMatrix`&&e.instanceMatrix&&(r=e.instanceMatrix),t===`instanceColor`&&e.instanceColor&&(r=e.instanceColor)),n===void 0||n.attribute!==r||r&&n.data!==r.data)return!0;s++}return a.attributesNum!==s||a.index!==r}function m(e,t,n,r){let i={},o=t.attributes,s=0,c=n.getAttributes();for(let t in c)if(c[t].location>=0){let n=o[t];n===void 0&&(t===`instanceMatrix`&&e.instanceMatrix&&(n=e.instanceMatrix),t===`instanceColor`&&e.instanceColor&&(n=e.instanceColor));let r={};r.attribute=n,n&&n.data&&(r.data=n.data),i[t]=r,s++}a.attributes=i,a.attributesNum=s,a.index=r}function h(){let e=a.newAttributes;for(let t=0,n=e.length;t<n;t++)e[t]=0}function g(e){_(e,0)}function _(t,n){let r=a.newAttributes,i=a.enabledAttributes,o=a.attributeDivisors;r[t]=1,i[t]===0&&(e.enableVertexAttribArray(t),i[t]=1),o[t]!==n&&(e.vertexAttribDivisor(t,n),o[t]=n)}function v(){let t=a.newAttributes,n=a.enabledAttributes;for(let r=0,i=n.length;r<i;r++)n[r]!==t[r]&&(e.disableVertexAttribArray(r),n[r]=0)}function y(t,n,r,i,a,o,s){s===!0?e.vertexAttribIPointer(t,n,r,a,o):e.vertexAttribPointer(t,n,r,i,a,o)}function b(n,r,i,a){h();let o=a.attributes,s=i.getAttributes(),c=r.defaultAttributeValues;for(let r in s){let i=s[r];if(i.location>=0){let s=o[r];if(s===void 0&&(r===`instanceMatrix`&&n.instanceMatrix&&(s=n.instanceMatrix),r===`instanceColor`&&n.instanceColor&&(s=n.instanceColor)),s!==void 0){let r=s.normalized,o=s.itemSize,c=t.get(s);if(c===void 0)continue;let l=c.buffer,u=c.type,d=c.bytesPerElement,f=u===e.INT||u===e.UNSIGNED_INT||s.gpuType===1013;if(s.isInterleavedBufferAttribute){let t=s.data,c=t.stride,p=s.offset;if(t.isInstancedInterleavedBuffer){for(let e=0;e<i.locationSize;e++)_(i.location+e,t.meshPerAttribute);n.isInstancedMesh!==!0&&a._maxInstanceCount===void 0&&(a._maxInstanceCount=t.meshPerAttribute*t.count)}else for(let e=0;e<i.locationSize;e++)g(i.location+e);e.bindBuffer(e.ARRAY_BUFFER,l);for(let e=0;e<i.locationSize;e++)y(i.location+e,o/i.locationSize,u,r,c*d,(p+o/i.locationSize*e)*d,f)}else{if(s.isInstancedBufferAttribute){for(let e=0;e<i.locationSize;e++)_(i.location+e,s.meshPerAttribute);n.isInstancedMesh!==!0&&a._maxInstanceCount===void 0&&(a._maxInstanceCount=s.meshPerAttribute*s.count)}else for(let e=0;e<i.locationSize;e++)g(i.location+e);e.bindBuffer(e.ARRAY_BUFFER,l);for(let e=0;e<i.locationSize;e++)y(i.location+e,o/i.locationSize,u,r,o*d,o/i.locationSize*e*d,f)}}else if(c!==void 0){let t=c[r];if(t!==void 0)switch(t.length){case 2:e.vertexAttrib2fv(i.location,t);break;case 3:e.vertexAttrib3fv(i.location,t);break;case 4:e.vertexAttrib4fv(i.location,t);break;default:e.vertexAttrib1fv(i.location,t)}}}}v()}function x(){T();for(let e in r){let t=r[e];for(let e in t){let n=t[e];for(let e in n){let t=n[e];for(let e in t)u(t[e].object),delete t[e];delete n[e]}}delete r[e]}}function S(e){if(r[e.id]===void 0)return;let t=r[e.id];for(let e in t){let n=t[e];for(let e in n){let t=n[e];for(let e in t)u(t[e].object),delete t[e];delete n[e]}}delete r[e.id]}function C(e){for(let t in r){let n=r[t];for(let t in n){let r=n[t];if(r[e.id]===void 0)continue;let i=r[e.id];for(let e in i)u(i[e].object),delete i[e];delete r[e.id]}}}function w(e){for(let t in r){let n=r[t],i=e.isInstancedMesh===!0?e.id:0,a=n[i];if(a!==void 0){for(let e in a){let t=a[e];for(let e in t)u(t[e].object),delete t[e];delete a[e]}delete n[i],Object.keys(n).length===0&&delete r[t]}}}function T(){E(),o=!0,a!==i&&(a=i,l(a.object))}function E(){i.geometry=null,i.program=null,i.wireframe=!1}return{setup:s,reset:T,resetDefaultState:E,dispose:x,releaseStatesOfGeometry:S,releaseStatesOfObject:w,releaseStatesOfProgram:C,initAttributes:h,enableAttribute:g,disableUnusedAttributes:v}}function mo(e,t,n){let r;function i(e){r=e}function a(t,i){e.drawArrays(r,t,i),n.update(i,r,1)}function o(t,i,a){a!==0&&(e.drawArraysInstanced(r,t,i,a),n.update(i,r,a))}function s(e,i,a){if(a===0)return;t.get(`WEBGL_multi_draw`).multiDrawArraysWEBGL(r,e,0,i,0,a);let o=0;for(let e=0;e<a;e++)o+=i[e];n.update(o,r,1)}this.setMode=i,this.render=a,this.renderInstances=o,this.renderMultiDraw=s}function ho(e,t,n,r){let i;function a(){if(i!==void 0)return i;if(t.has(`EXT_texture_filter_anisotropic`)===!0){let n=t.get(`EXT_texture_filter_anisotropic`);i=e.getParameter(n.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else i=0;return i}function o(t){return!(t!==1023&&r.convert(t)!==e.getParameter(e.IMPLEMENTATION_COLOR_READ_FORMAT))}function s(n){let i=n===1016&&(t.has(`EXT_color_buffer_half_float`)||t.has(`EXT_color_buffer_float`));return!(n!==1009&&r.convert(n)!==e.getParameter(e.IMPLEMENTATION_COLOR_READ_TYPE)&&n!==1015&&!i)}function c(t){if(t===`highp`){if(e.getShaderPrecisionFormat(e.VERTEX_SHADER,e.HIGH_FLOAT).precision>0&&e.getShaderPrecisionFormat(e.FRAGMENT_SHADER,e.HIGH_FLOAT).precision>0)return`highp`;t=`mediump`}return t===`mediump`&&e.getShaderPrecisionFormat(e.VERTEX_SHADER,e.MEDIUM_FLOAT).precision>0&&e.getShaderPrecisionFormat(e.FRAGMENT_SHADER,e.MEDIUM_FLOAT).precision>0?`mediump`:`lowp`}let l=n.precision===void 0?`highp`:n.precision,u=c(l);u!==l&&(W(`WebGLRenderer:`,l,`not supported, using`,u,`instead.`),l=u);let d=n.logarithmicDepthBuffer===!0,f=n.reversedDepthBuffer===!0&&t.has(`EXT_clip_control`);n.reversedDepthBuffer===!0&&f===!1&&W(`WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.`);let p=e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS),m=e.getParameter(e.MAX_VERTEX_TEXTURE_IMAGE_UNITS),h=e.getParameter(e.MAX_TEXTURE_SIZE),g=e.getParameter(e.MAX_CUBE_MAP_TEXTURE_SIZE),_=e.getParameter(e.MAX_VERTEX_ATTRIBS),v=e.getParameter(e.MAX_VERTEX_UNIFORM_VECTORS),y=e.getParameter(e.MAX_VARYING_VECTORS),b=e.getParameter(e.MAX_FRAGMENT_UNIFORM_VECTORS),x=e.getParameter(e.MAX_SAMPLES),S=e.getParameter(e.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:a,getMaxPrecision:c,textureFormatReadable:o,textureTypeReadable:s,precision:l,logarithmicDepthBuffer:d,reversedDepthBuffer:f,maxTextures:p,maxVertexTextures:m,maxTextureSize:h,maxCubemapSize:g,maxAttributes:_,maxVertexUniforms:v,maxVaryings:y,maxFragmentUniforms:b,maxSamples:x,samples:S}}function go(e){let t=this,n=null,r=0,i=!1,a=!1,o=new ki,s=new J,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(e,t){let n=e.length!==0||t||r!==0||i;return i=t,r=e.length,n},this.beginShadows=function(){a=!0,u(null)},this.endShadows=function(){a=!1},this.setGlobalState=function(e,t){n=u(e,t,0)},this.setState=function(t,o,s){let d=t.clippingPlanes,f=t.clipIntersection,p=t.clipShadows,m=e.get(t);if(!i||d===null||d.length===0||a&&!p)a?u(null):l();else{let e=a?0:r,t=e*4,i=m.clippingState||null;c.value=i,i=u(d,o,t,s);for(let e=0;e!==t;++e)i[e]=n[e];m.clippingState=i,this.numIntersection=f?this.numPlanes:0,this.numPlanes+=e}};function l(){c.value!==n&&(c.value=n,c.needsUpdate=r>0),t.numPlanes=r,t.numIntersection=0}function u(e,n,r,i){let a=e===null?0:e.length,l=null;if(a!==0){if(l=c.value,i!==!0||l===null){let t=r+a*4,i=n.matrixWorldInverse;s.getNormalMatrix(i),(l===null||l.length<t)&&(l=new Float32Array(t));for(let t=0,n=r;t!==a;++t,n+=4)o.copy(e[t]).applyMatrix4(i,s),o.normal.toArray(l,n),l[n+3]=o.constant}c.value=l,c.needsUpdate=!0}return t.numPlanes=a,t.numIntersection=0,l}}var _o=4,vo=[.125,.215,.35,.446,.526,.582],yo=20,bo=256,xo=new La,So=new Ln,Co=null,wo=0,To=0,Eo=!1,Do=new q,Oo=class{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,r=100,i={}){let{size:a=256,position:o=Do}=i;Co=this._renderer.getRenderTarget(),wo=this._renderer.getActiveCubeFace(),To=this._renderer.getActiveMipmapLevel(),Eo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);let s=this._allocateTargets();return s.depthBuffer=!0,this._sceneToCubeUV(e,n,r,s,o),t>0&&this._blur(s,0,0,t),this._applyPMREM(s),this._cleanup(s),s}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Fo(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Po(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=2**this._lodMax}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Co,wo,To),this._renderer.xr.enabled=Eo,e.scissorTest=!1,jo(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===301||e.mapping===302?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Co=this._renderer.getRenderTarget(),wo=this._renderer.getActiveCubeFace(),To=this._renderer.getActiveMipmapLevel(),Eo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){let e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:N,minFilter:N,generateMipmaps:!1,type:ie,format:ue,colorSpace:ot,depthBuffer:!1},r=Ao(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Ao(e,t,n);let{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=ko(r)),this._blurMaterial=No(r,e,t),this._ggxMaterial=Mo(r,e,t)}return r}_compileMaterial(e){let t=new Si(new Pr,e);this._renderer.compile(t,xo)}_sceneToCubeUV(e,t,n,r,i){let a=new Ia(90,1,t,n),o=[1,-1,1,1,1,1],s=[1,1,1,-1,-1,-1],c=this._renderer,l=c.autoClear,u=c.toneMapping;c.getClearColor(So),c.toneMapping=0,c.autoClear=!1,c.state.buffers.depth.getReversed()&&(c.setRenderTarget(r),c.clearDepth(),c.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Si(new Ki,new ui({name:`PMREM.Background`,side:1,depthWrite:!1,depthTest:!1})));let d=this._backgroundBox,f=d.material,p=!1,m=e.background;m?m.isColor&&(f.color.copy(m),e.background=null,p=!0):(f.color.copy(So),p=!0);for(let t=0;t<6;t++){let n=t%3;n===0?(a.up.set(0,o[t],0),a.position.set(i.x,i.y,i.z),a.lookAt(i.x+s[t],i.y,i.z)):n===1?(a.up.set(0,0,o[t]),a.position.set(i.x,i.y,i.z),a.lookAt(i.x,i.y+s[t],i.z)):(a.up.set(0,o[t],0),a.position.set(i.x,i.y,i.z),a.lookAt(i.x,i.y,i.z+s[t]));let l=this._cubeSize;jo(r,n*l,t>2?l:0,l,l),c.setRenderTarget(r),p&&c.render(d,a),c.render(e,a)}c.toneMapping=u,c.autoClear=l,e.background=m}_textureToCubeUV(e,t){let n=this._renderer,r=e.mapping===301||e.mapping===302;r?(this._cubemapMaterial===null&&(this._cubemapMaterial=Fo()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Po());let i=r?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=i;let o=i.uniforms;o.envMap.value=e;let s=this._cubeSize;jo(t,0,0,3*s,2*s),n.setRenderTarget(t),n.render(a,xo)}_applyPMREM(e){let t=this._renderer,n=t.autoClear;t.autoClear=!1;let r=this._lodMeshes.length;for(let t=1;t<r;t++)this._applyGGXFilter(e,t-1,t);t.autoClear=n}_applyGGXFilter(e,t,n){let r=this._renderer,i=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[n];o.material=a;let s=a.uniforms,c=n/(this._lodMeshes.length-1),l=t/(this._lodMeshes.length-1),u=Math.sqrt(c*c-l*l)*(0+c*1.25),{_lodMax:d}=this,f=this._sizeLods[n],p=3*f*(n>d-_o?n-d+_o:0),m=4*(this._cubeSize-f);s.envMap.value=e.texture,s.roughness.value=u,s.mipInt.value=d-t,jo(i,p,m,3*f,2*f),r.setRenderTarget(i),r.render(o,xo),s.envMap.value=i.texture,s.roughness.value=0,s.mipInt.value=d-n,jo(e,p,m,3*f,2*f),r.setRenderTarget(e),r.render(o,xo)}_blur(e,t,n,r,i){let a=this._pingPongRenderTarget;this._halfBlur(e,a,t,n,r,`latitudinal`,i),this._halfBlur(a,e,n,n,r,`longitudinal`,i)}_halfBlur(e,t,n,r,i,a,o){let s=this._renderer,c=this._blurMaterial;a!==`latitudinal`&&a!==`longitudinal`&&G(`blur direction must be either latitudinal or longitudinal!`);let l=this._lodMeshes[r];l.material=c;let u=c.uniforms,d=this._sizeLods[n]-1,f=isFinite(i)?Math.PI/(2*d):2*Math.PI/(2*yo-1),p=i/f,m=isFinite(i)?1+Math.floor(3*p):yo;m>yo&&W(`sigmaRadians, ${i}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${yo}`);let h=[],g=0;for(let e=0;e<yo;++e){let t=e/p,n=Math.exp(-t*t/2);h.push(n),e===0?g+=n:e<m&&(g+=2*n)}for(let e=0;e<h.length;e++)h[e]=h[e]/g;u.envMap.value=e.texture,u.samples.value=m,u.weights.value=h,u.latitudinal.value=a===`latitudinal`,o&&(u.poleAxis.value=o);let{_lodMax:_}=this;u.dTheta.value=f,u.mipInt.value=_-n;let v=this._sizeLods[r];jo(t,3*v*(r>_-_o?r-_+_o:0),4*(this._cubeSize-v),3*v,2*v),s.setRenderTarget(t),s.render(l,xo)}};function ko(e){let t=[],n=[],r=[],i=e,a=e-_o+1+vo.length;for(let o=0;o<a;o++){let a=2**i;t.push(a);let s=1/a;o>e-_o?s=vo[o-e+_o-1]:o===0&&(s=0),n.push(s);let c=1/(a-2),l=-c,u=1+c,d=[l,l,u,l,u,u,l,l,u,u,l,u],f=new Float32Array(108),p=new Float32Array(72),m=new Float32Array(36);for(let e=0;e<6;e++){let t=e%3*2/3-1,n=e>2?0:-1,r=[t,n,0,t+2/3,n,0,t+2/3,n+1,0,t,n,0,t+2/3,n+1,0,t,n+1,0];f.set(r,18*e),p.set(d,12*e);let i=[e,e,e,e,e,e];m.set(i,6*e)}let h=new Pr;h.setAttribute(`position`,new yr(f,3)),h.setAttribute(`uv`,new yr(p,2)),h.setAttribute(`faceIndex`,new yr(m,1)),r.push(new Si(h,null)),i>_o&&i--}return{lodMeshes:r,sizeLods:t,sigmas:n}}function Ao(e,t,n){let r=new Qt(e,t,n);return r.texture.mapping=306,r.texture.name=`PMREM.cubeUv`,r.scissorTest=!0,r}function jo(e,t,n,r,i){e.viewport.set(t,n,r,i),e.scissor.set(t,n,r,i)}function Mo(e,t,n){return new na({name:`PMREMGGXConvolution`,defines:{GGX_SAMPLES:bo,CUBEUV_TEXEL_WIDTH:1/t,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${e}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Io(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:0,depthTest:!1,depthWrite:!1})}function No(e,t,n){let r=new Float32Array(yo),i=new q(0,1,0);return new na({name:`SphericalGaussianBlur`,defines:{n:yo,CUBEUV_TEXEL_WIDTH:1/t,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${e}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:r},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:i}},vertexShader:Io(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:0,depthTest:!1,depthWrite:!1})}function Po(){return new na({name:`EquirectangularToCubeUV`,uniforms:{envMap:{value:null}},vertexShader:Io(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:0,depthTest:!1,depthWrite:!1})}function Fo(){return new na({name:`CubemapToCubeUV`,uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Io(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:0,depthTest:!1,depthWrite:!1})}function Io(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}var Lo=class extends Qt{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;let n={width:e,height:e,depth:1};this.texture=new Vi([n,n,n,n,n,n]),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;let n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},r=new Ki(5,5,5),i=new na({name:`CubemapFromEquirect`,uniforms:Ji(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:1,blending:0});i.uniforms.tEquirect.value=t;let a=new Si(r,i),o=t.minFilter;return t.minFilter===1008&&(t.minFilter=N),new Ha(1,10,this).update(e,a),t.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,t=!0,n=!0,r=!0){let i=e.getRenderTarget();for(let i=0;i<6;i++)e.setRenderTarget(this,i),e.clear(t,n,r);e.setRenderTarget(i)}};function Ro(e){let t=new WeakMap,n=new WeakMap,r=null;function i(e,t=!1){return e==null?null:t?o(e):a(e)}function a(n){if(n&&n.isTexture){let r=n.mapping;if(r===303||r===304)if(t.has(n)){let e=t.get(n).texture;return s(e,n.mapping)}else{let r=n.image;if(r&&r.height>0){let i=new Lo(r.height);return i.fromEquirectangularTexture(e,n),t.set(n,i),n.addEventListener(`dispose`,l),s(i.texture,n.mapping)}else return null}}return n}function o(t){if(t&&t.isTexture){let i=t.mapping,a=i===303||i===304,o=i===301||i===302;if(a||o){let i=n.get(t),s=i===void 0?0:i.texture.pmremVersion;if(t.isRenderTargetTexture&&t.pmremVersion!==s)return r===null&&(r=new Oo(e)),i=a?r.fromEquirectangular(t,i):r.fromCubemap(t,i),i.texture.pmremVersion=t.pmremVersion,n.set(t,i),i.texture;if(i!==void 0)return i.texture;{let s=t.image;return a&&s&&s.height>0||o&&s&&c(s)?(r===null&&(r=new Oo(e)),i=a?r.fromEquirectangular(t):r.fromCubemap(t),i.texture.pmremVersion=t.pmremVersion,n.set(t,i),t.addEventListener(`dispose`,u),i.texture):null}}}return t}function s(e,t){return t===303?e.mapping=301:t===304&&(e.mapping=302),e}function c(e){let t=0;for(let n=0;n<6;n++)e[n]!==void 0&&t++;return t===6}function l(e){let n=e.target;n.removeEventListener(`dispose`,l);let r=t.get(n);r!==void 0&&(t.delete(n),r.dispose())}function u(e){let t=e.target;t.removeEventListener(`dispose`,u);let r=n.get(t);r!==void 0&&(n.delete(t),r.dispose())}function d(){t=new WeakMap,n=new WeakMap,r!==null&&(r.dispose(),r=null)}return{get:i,dispose:d}}function zo(e){let t={};function n(n){if(t[n]!==void 0)return t[n];let r=e.getExtension(n);return t[n]=r,r}return{has:function(e){return n(e)!==null},init:function(){n(`EXT_color_buffer_float`),n(`WEBGL_clip_cull_distance`),n(`OES_texture_float_linear`),n(`EXT_color_buffer_half_float`),n(`WEBGL_multisampled_render_to_texture`),n(`WEBGL_render_shared_exponent`)},get:function(e){let t=n(e);return t===null&&bt(`WebGLRenderer: `+e+` extension not supported.`),t}}}function Bo(e,t,n,r){let i={},a=new WeakMap;function o(e){let s=e.target;s.index!==null&&t.remove(s.index);for(let e in s.attributes)t.remove(s.attributes[e]);s.removeEventListener(`dispose`,o),delete i[s.id];let c=a.get(s);c&&(t.remove(c),a.delete(s)),r.releaseStatesOfGeometry(s),s.isInstancedBufferGeometry===!0&&delete s._maxInstanceCount,n.memory.geometries--}function s(e,t){return i[t.id]===!0?t:(t.addEventListener(`dispose`,o),i[t.id]=!0,n.memory.geometries++,t)}function c(n){let r=n.attributes;for(let n in r)t.update(r[n],e.ARRAY_BUFFER)}function l(e){let n=[],r=e.index,i=e.attributes.position,o=0;if(i===void 0)return;if(r!==null){let e=r.array;o=r.version;for(let t=0,r=e.length;t<r;t+=3){let r=e[t+0],i=e[t+1],a=e[t+2];n.push(r,i,i,a,a,r)}}else{let e=i.array;o=i.version;for(let t=0,r=e.length/3-1;t<r;t+=3){let e=t+0,r=t+1,i=t+2;n.push(e,r,r,i,i,e)}}let s=new(i.count>=65535?xr:br)(n,1);s.version=o;let c=a.get(e);c&&t.remove(c),a.set(e,s)}function u(e){let t=a.get(e);if(t){let n=e.index;n!==null&&t.version<n.version&&l(e)}else l(e);return a.get(e)}return{get:s,update:c,getWireframeAttribute:u}}function Vo(e,t,n){let r;function i(e){r=e}let a,o;function s(e){a=e.type,o=e.bytesPerElement}function c(t,i){e.drawElements(r,i,a,t*o),n.update(i,r,1)}function l(t,i,s){s!==0&&(e.drawElementsInstanced(r,i,a,t*o,s),n.update(i,r,s))}function u(e,i,o){if(o===0)return;t.get(`WEBGL_multi_draw`).multiDrawElementsWEBGL(r,i,0,a,e,0,o);let s=0;for(let e=0;e<o;e++)s+=i[e];n.update(s,r,1)}this.setMode=i,this.setIndex=s,this.render=c,this.renderInstances=l,this.renderMultiDraw=u}function Ho(e){let t={geometries:0,textures:0},n={frame:0,calls:0,triangles:0,points:0,lines:0};function r(t,r,i){switch(n.calls++,r){case e.TRIANGLES:n.triangles+=t/3*i;break;case e.LINES:n.lines+=t/2*i;break;case e.LINE_STRIP:n.lines+=i*(t-1);break;case e.LINE_LOOP:n.lines+=i*t;break;case e.POINTS:n.points+=i*t;break;default:G(`WebGLInfo: Unknown draw mode:`,r);break}}function i(){n.calls=0,n.triangles=0,n.points=0,n.lines=0}return{memory:t,render:n,programs:null,autoReset:!0,reset:i,update:r}}function Uo(e,t,n){let r=new WeakMap,i=new Xt;function a(a,o,s){let c=a.morphTargetInfluences,l=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,u=l===void 0?0:l.length,d=r.get(o);if(d===void 0||d.count!==u){d!==void 0&&d.texture.dispose();let e=o.morphAttributes.position!==void 0,n=o.morphAttributes.normal!==void 0,a=o.morphAttributes.color!==void 0,s=o.morphAttributes.position||[],c=o.morphAttributes.normal||[],l=o.morphAttributes.color||[],f=0;e===!0&&(f=1),n===!0&&(f=2),a===!0&&(f=3);let p=o.attributes.position.count*f,m=1;p>t.maxTextureSize&&(m=Math.ceil(p/t.maxTextureSize),p=t.maxTextureSize);let h=new Float32Array(p*m*4*u),g=new $t(h,p,m,u);g.type=re,g.needsUpdate=!0;let _=f*4;for(let t=0;t<u;t++){let r=s[t],o=c[t],u=l[t],d=p*m*4*t;for(let t=0;t<r.count;t++){let s=t*_;e===!0&&(i.fromBufferAttribute(r,t),h[d+s+0]=i.x,h[d+s+1]=i.y,h[d+s+2]=i.z,h[d+s+3]=0),n===!0&&(i.fromBufferAttribute(o,t),h[d+s+4]=i.x,h[d+s+5]=i.y,h[d+s+6]=i.z,h[d+s+7]=0),a===!0&&(i.fromBufferAttribute(u,t),h[d+s+8]=i.x,h[d+s+9]=i.y,h[d+s+10]=i.z,h[d+s+11]=u.itemSize===4?i.w:1)}}d={count:u,texture:g,size:new Mt(p,m)},r.set(o,d);function v(){g.dispose(),r.delete(o),o.removeEventListener(`dispose`,v)}o.addEventListener(`dispose`,v)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)s.getUniforms().setValue(e,`morphTexture`,a.morphTexture,n);else{let t=0;for(let e=0;e<c.length;e++)t+=c[e];let n=o.morphTargetsRelative?1:1-t;s.getUniforms().setValue(e,`morphTargetBaseInfluence`,n),s.getUniforms().setValue(e,`morphTargetInfluences`,c)}s.getUniforms().setValue(e,`morphTargetsTexture`,d.texture,n),s.getUniforms().setValue(e,`morphTargetsTextureSize`,d.size)}return{update:a}}function Wo(e,t,n,r,i){let a=new WeakMap;function o(r){let o=i.render.frame,s=r.geometry,l=t.get(r,s);if(a.get(l)!==o&&(t.update(l),a.set(l,o)),r.isInstancedMesh&&(r.hasEventListener(`dispose`,c)===!1&&r.addEventListener(`dispose`,c),a.get(r)!==o&&(n.update(r.instanceMatrix,e.ARRAY_BUFFER),r.instanceColor!==null&&n.update(r.instanceColor,e.ARRAY_BUFFER),a.set(r,o))),r.isSkinnedMesh){let e=r.skeleton;a.get(e)!==o&&(e.update(),a.set(e,o))}return l}function s(){a=new WeakMap}function c(e){let t=e.target;t.removeEventListener(`dispose`,c),r.releaseStatesOfObject(t),n.remove(t.instanceMatrix),t.instanceColor!==null&&n.remove(t.instanceColor)}return{update:o,dispose:s}}var Go={1:`LINEAR_TONE_MAPPING`,2:`REINHARD_TONE_MAPPING`,3:`CINEON_TONE_MAPPING`,4:`ACES_FILMIC_TONE_MAPPING`,6:`AGX_TONE_MAPPING`,7:`NEUTRAL_TONE_MAPPING`,5:`CUSTOM_TONE_MAPPING`};function Ko(e,t,n,r,i,a){let o=new Qt(t,n,{type:e,depthBuffer:i,stencilBuffer:a,samples:r?4:0,depthTexture:i?new Ui(t,n):void 0}),s=new Qt(t,n,{type:ie,depthBuffer:!1,stencilBuffer:!1}),c=new Pr;c.setAttribute(`position`,new Sr([-1,3,0,-1,-1,0,3,-1,0],3)),c.setAttribute(`uv`,new Sr([0,2,0,0,2,0],2));let l=new ra({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),u=new Si(c,l),d=new La(-1,1,1,-1,0,1),f=null,p=null,m=!1,h,g=null,_=[],v=!1;this.setSize=function(e,t){o.setSize(e,t),s.setSize(e,t);for(let n=0;n<_.length;n++){let r=_[n];r.setSize&&r.setSize(e,t)}},this.setEffects=function(e){_=e,v=_.length>0&&_[0].isRenderPass===!0;let t=o.width,n=o.height;for(let e=0;e<_.length;e++){let r=_[e];r.setSize&&r.setSize(t,n)}},this.begin=function(e,t){if(m||e.toneMapping===0&&_.length===0)return!1;if(g=t,t!==null){let e=t.width,n=t.height;(o.width!==e||o.height!==n)&&this.setSize(e,n)}return v===!1&&e.setRenderTarget(o),h=e.toneMapping,e.toneMapping=0,!0},this.hasRenderPass=function(){return v},this.end=function(e,t){e.toneMapping=h,m=!0;let n=o,r=s;for(let i=0;i<_.length;i++){let a=_[i];if(a.enabled!==!1&&(a.render(e,r,n,t),a.needsSwap!==!1)){let e=n;n=r,r=e}}if(f!==e.outputColorSpace||p!==e.toneMapping){f=e.outputColorSpace,p=e.toneMapping,l.defines={},Y.getTransfer(f)===`srgb`&&(l.defines.SRGB_TRANSFER=``);let t=Go[p];t&&(l.defines[t]=``),l.needsUpdate=!0}l.uniforms.tDiffuse.value=n.texture,e.setRenderTarget(g),e.render(u,d),g=null,m=!1},this.isCompositing=function(){return m},this.dispose=function(){o.depthTexture&&o.depthTexture.dispose(),o.dispose(),s.dispose(),c.dispose(),l.dispose()}}var qo=new Yt,Jo=new Ui(1,1),Yo=new $t,Xo=new en,Zo=new Vi,Qo=[],$o=[],es=new Float32Array(16),ts=new Float32Array(9),ns=new Float32Array(4);function rs(e,t,n){let r=e[0];if(r<=0||r>0)return e;let i=t*n,a=Qo[i];if(a===void 0&&(a=new Float32Array(i),Qo[i]=a),t!==0){r.toArray(a,0);for(let r=1,i=0;r!==t;++r)i+=n,e[r].toArray(a,i)}return a}function is(e,t){if(e.length!==t.length)return!1;for(let n=0,r=e.length;n<r;n++)if(e[n]!==t[n])return!1;return!0}function as(e,t){for(let n=0,r=t.length;n<r;n++)e[n]=t[n]}function os(e,t){let n=$o[t];n===void 0&&(n=new Int32Array(t),$o[t]=n);for(let r=0;r!==t;++r)n[r]=e.allocateTextureUnit();return n}function ss(e,t){let n=this.cache;n[0]!==t&&(e.uniform1f(this.addr,t),n[0]=t)}function cs(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y)&&(e.uniform2f(this.addr,t.x,t.y),n[0]=t.x,n[1]=t.y);else{if(is(n,t))return;e.uniform2fv(this.addr,t),as(n,t)}}function ls(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y||n[2]!==t.z)&&(e.uniform3f(this.addr,t.x,t.y,t.z),n[0]=t.x,n[1]=t.y,n[2]=t.z);else if(t.r!==void 0)(n[0]!==t.r||n[1]!==t.g||n[2]!==t.b)&&(e.uniform3f(this.addr,t.r,t.g,t.b),n[0]=t.r,n[1]=t.g,n[2]=t.b);else{if(is(n,t))return;e.uniform3fv(this.addr,t),as(n,t)}}function us(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y||n[2]!==t.z||n[3]!==t.w)&&(e.uniform4f(this.addr,t.x,t.y,t.z,t.w),n[0]=t.x,n[1]=t.y,n[2]=t.z,n[3]=t.w);else{if(is(n,t))return;e.uniform4fv(this.addr,t),as(n,t)}}function ds(e,t){let n=this.cache,r=t.elements;if(r===void 0){if(is(n,t))return;e.uniformMatrix2fv(this.addr,!1,t),as(n,t)}else{if(is(n,r))return;ns.set(r),e.uniformMatrix2fv(this.addr,!1,ns),as(n,r)}}function fs(e,t){let n=this.cache,r=t.elements;if(r===void 0){if(is(n,t))return;e.uniformMatrix3fv(this.addr,!1,t),as(n,t)}else{if(is(n,r))return;ts.set(r),e.uniformMatrix3fv(this.addr,!1,ts),as(n,r)}}function ps(e,t){let n=this.cache,r=t.elements;if(r===void 0){if(is(n,t))return;e.uniformMatrix4fv(this.addr,!1,t),as(n,t)}else{if(is(n,r))return;es.set(r),e.uniformMatrix4fv(this.addr,!1,es),as(n,r)}}function ms(e,t){let n=this.cache;n[0]!==t&&(e.uniform1i(this.addr,t),n[0]=t)}function hs(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y)&&(e.uniform2i(this.addr,t.x,t.y),n[0]=t.x,n[1]=t.y);else{if(is(n,t))return;e.uniform2iv(this.addr,t),as(n,t)}}function gs(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y||n[2]!==t.z)&&(e.uniform3i(this.addr,t.x,t.y,t.z),n[0]=t.x,n[1]=t.y,n[2]=t.z);else{if(is(n,t))return;e.uniform3iv(this.addr,t),as(n,t)}}function _s(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y||n[2]!==t.z||n[3]!==t.w)&&(e.uniform4i(this.addr,t.x,t.y,t.z,t.w),n[0]=t.x,n[1]=t.y,n[2]=t.z,n[3]=t.w);else{if(is(n,t))return;e.uniform4iv(this.addr,t),as(n,t)}}function vs(e,t){let n=this.cache;n[0]!==t&&(e.uniform1ui(this.addr,t),n[0]=t)}function ys(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y)&&(e.uniform2ui(this.addr,t.x,t.y),n[0]=t.x,n[1]=t.y);else{if(is(n,t))return;e.uniform2uiv(this.addr,t),as(n,t)}}function bs(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y||n[2]!==t.z)&&(e.uniform3ui(this.addr,t.x,t.y,t.z),n[0]=t.x,n[1]=t.y,n[2]=t.z);else{if(is(n,t))return;e.uniform3uiv(this.addr,t),as(n,t)}}function xs(e,t){let n=this.cache;if(t.x!==void 0)(n[0]!==t.x||n[1]!==t.y||n[2]!==t.z||n[3]!==t.w)&&(e.uniform4ui(this.addr,t.x,t.y,t.z,t.w),n[0]=t.x,n[1]=t.y,n[2]=t.z,n[3]=t.w);else{if(is(n,t))return;e.uniform4uiv(this.addr,t),as(n,t)}}function Ss(e,t,n){let r=this.cache,i=n.allocateTextureUnit();r[0]!==i&&(e.uniform1i(this.addr,i),r[0]=i);let a;this.type===e.SAMPLER_2D_SHADOW?(Jo.compareFunction=n.isReversedDepthBuffer()?518:515,a=Jo):a=qo,n.setTexture2D(t||a,i)}function Cs(e,t,n){let r=this.cache,i=n.allocateTextureUnit();r[0]!==i&&(e.uniform1i(this.addr,i),r[0]=i),n.setTexture3D(t||Xo,i)}function ws(e,t,n){let r=this.cache,i=n.allocateTextureUnit();r[0]!==i&&(e.uniform1i(this.addr,i),r[0]=i),n.setTextureCube(t||Zo,i)}function Ts(e,t,n){let r=this.cache,i=n.allocateTextureUnit();r[0]!==i&&(e.uniform1i(this.addr,i),r[0]=i),n.setTexture2DArray(t||Yo,i)}function Es(e){switch(e){case 5126:return ss;case 35664:return cs;case 35665:return ls;case 35666:return us;case 35674:return ds;case 35675:return fs;case 35676:return ps;case 5124:case 35670:return ms;case 35667:case 35671:return hs;case 35668:case 35672:return gs;case 35669:case 35673:return _s;case 5125:return vs;case 36294:return ys;case 36295:return bs;case 36296:return xs;case 35678:case 36198:case 36298:case 36306:case 35682:return Ss;case 35679:case 36299:case 36307:return Cs;case 35680:case 36300:case 36308:case 36293:return ws;case 36289:case 36303:case 36311:case 36292:return Ts}}function Ds(e,t){e.uniform1fv(this.addr,t)}function Os(e,t){let n=rs(t,this.size,2);e.uniform2fv(this.addr,n)}function ks(e,t){let n=rs(t,this.size,3);e.uniform3fv(this.addr,n)}function As(e,t){let n=rs(t,this.size,4);e.uniform4fv(this.addr,n)}function js(e,t){let n=rs(t,this.size,4);e.uniformMatrix2fv(this.addr,!1,n)}function Ms(e,t){let n=rs(t,this.size,9);e.uniformMatrix3fv(this.addr,!1,n)}function Ns(e,t){let n=rs(t,this.size,16);e.uniformMatrix4fv(this.addr,!1,n)}function Ps(e,t){e.uniform1iv(this.addr,t)}function Fs(e,t){e.uniform2iv(this.addr,t)}function Is(e,t){e.uniform3iv(this.addr,t)}function Ls(e,t){e.uniform4iv(this.addr,t)}function Rs(e,t){e.uniform1uiv(this.addr,t)}function zs(e,t){e.uniform2uiv(this.addr,t)}function Bs(e,t){e.uniform3uiv(this.addr,t)}function Vs(e,t){e.uniform4uiv(this.addr,t)}function Hs(e,t,n){let r=this.cache,i=t.length,a=os(n,i);is(r,a)||(e.uniform1iv(this.addr,a),as(r,a));let o;o=this.type===e.SAMPLER_2D_SHADOW?Jo:qo;for(let e=0;e!==i;++e)n.setTexture2D(t[e]||o,a[e])}function Us(e,t,n){let r=this.cache,i=t.length,a=os(n,i);is(r,a)||(e.uniform1iv(this.addr,a),as(r,a));for(let e=0;e!==i;++e)n.setTexture3D(t[e]||Xo,a[e])}function Ws(e,t,n){let r=this.cache,i=t.length,a=os(n,i);is(r,a)||(e.uniform1iv(this.addr,a),as(r,a));for(let e=0;e!==i;++e)n.setTextureCube(t[e]||Zo,a[e])}function Gs(e,t,n){let r=this.cache,i=t.length,a=os(n,i);is(r,a)||(e.uniform1iv(this.addr,a),as(r,a));for(let e=0;e!==i;++e)n.setTexture2DArray(t[e]||Yo,a[e])}function Ks(e){switch(e){case 5126:return Ds;case 35664:return Os;case 35665:return ks;case 35666:return As;case 35674:return js;case 35675:return Ms;case 35676:return Ns;case 5124:case 35670:return Ps;case 35667:case 35671:return Fs;case 35668:case 35672:return Is;case 35669:case 35673:return Ls;case 5125:return Rs;case 36294:return zs;case 36295:return Bs;case 36296:return Vs;case 35678:case 36198:case 36298:case 36306:case 35682:return Hs;case 35679:case 36299:case 36307:return Us;case 35680:case 36300:case 36308:case 36293:return Ws;case 36289:case 36303:case 36311:case 36292:return Gs}}var qs=class{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=Es(t.type)}},Js=class{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=Ks(t.type)}},Ys=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){let r=this.seq;for(let i=0,a=r.length;i!==a;++i){let a=r[i];a.setValue(e,t[a.id],n)}}},Xs=/(\w+)(\])?(\[|\.)?/g;function Zs(e,t){e.seq.push(t),e.map[t.id]=t}function Qs(e,t,n){let r=e.name,i=r.length;for(Xs.lastIndex=0;;){let a=Xs.exec(r),o=Xs.lastIndex,s=a[1],c=a[2]===`]`,l=a[3];if(c&&(s|=0),l===void 0||l===`[`&&o+2===i){Zs(n,l===void 0?new qs(s,e,t):new Js(s,e,t));break}else{let e=n.map[s];e===void 0&&(e=new Ys(s),Zs(n,e)),n=e}}}var $s=class{constructor(e,t){this.seq=[],this.map={};let n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let r=0;r<n;++r){let n=e.getActiveUniform(t,r);Qs(n,e.getUniformLocation(t,n.name),this)}let r=[],i=[];for(let t of this.seq)t.type===e.SAMPLER_2D_SHADOW||t.type===e.SAMPLER_CUBE_SHADOW||t.type===e.SAMPLER_2D_ARRAY_SHADOW?r.push(t):i.push(t);r.length>0&&(this.seq=r.concat(i))}setValue(e,t,n,r){let i=this.map[t];i!==void 0&&i.setValue(e,n,r)}setOptional(e,t,n){let r=t[n];r!==void 0&&this.setValue(e,n,r)}static upload(e,t,n,r){for(let i=0,a=t.length;i!==a;++i){let a=t[i],o=n[a.id];o.needsUpdate!==!1&&a.setValue(e,o.value,r)}}static seqWithValue(e,t){let n=[];for(let r=0,i=e.length;r!==i;++r){let i=e[r];i.id in t&&n.push(i)}return n}};function ec(e,t,n){let r=e.createShader(t);return e.shaderSource(r,n),e.compileShader(r),r}var tc=37297,nc=0;function rc(e,t){let n=e.split(`
`),r=[],i=Math.max(t-6,0),a=Math.min(t+6,n.length);for(let e=i;e<a;e++){let i=e+1;r.push(`${i===t?`>`:` `} ${i}: ${n[e]}`)}return r.join(`
`)}var ic=new J;function ac(e){Y._getMatrix(ic,Y.workingColorSpace,e);let t=`mat3( ${ic.elements.map(e=>e.toFixed(4))} )`;switch(Y.getTransfer(e)){case st:return[t,`LinearTransferOETF`];case ct:return[t,`sRGBTransferOETF`];default:return W(`WebGLProgram: Unsupported color space: `,e),[t,`LinearTransferOETF`]}}function oc(e,t,n){let r=e.getShaderParameter(t,e.COMPILE_STATUS),i=(e.getShaderInfoLog(t)||``).trim();if(r&&i===``)return``;let a=/ERROR: 0:(\d+)/.exec(i);if(a){let r=parseInt(a[1]);return n.toUpperCase()+`

`+i+`

`+rc(e.getShaderSource(t),r)}else return i}function sc(e,t){let n=ac(t);return[`vec4 ${e}( vec4 value ) {`,`	return ${n[1]}( vec4( value.rgb * ${n[0]}, value.a ) );`,`}`].join(`
`)}var cc={1:`Linear`,2:`Reinhard`,3:`Cineon`,4:`ACESFilmic`,6:`AgX`,7:`Neutral`,5:`Custom`};function lc(e,t){let n=cc[t];return n===void 0?(W(`WebGLProgram: Unsupported toneMapping:`,t),`vec3 `+e+`( vec3 color ) { return LinearToneMapping( color ); }`):`vec3 `+e+`( vec3 color ) { return `+n+`ToneMapping( color ); }`}var uc=new q;function dc(){return Y.getLuminanceCoefficients(uc),[`float luminance( const in vec3 rgb ) {`,`	const vec3 weights = vec3( ${uc.x.toFixed(4)}, ${uc.y.toFixed(4)}, ${uc.z.toFixed(4)} );`,`	return dot( weights, rgb );`,`}`].join(`
`)}function fc(e){return[e.extensionClipCullDistance?`#extension GL_ANGLE_clip_cull_distance : require`:``,e.extensionMultiDraw?`#extension GL_ANGLE_multi_draw : require`:``].filter(hc).join(`
`)}function pc(e){let t=[];for(let n in e){let r=e[n];r!==!1&&t.push(`#define `+n+` `+r)}return t.join(`
`)}function mc(e,t){let n={},r=e.getProgramParameter(t,e.ACTIVE_ATTRIBUTES);for(let i=0;i<r;i++){let r=e.getActiveAttrib(t,i),a=r.name,o=1;r.type===e.FLOAT_MAT2&&(o=2),r.type===e.FLOAT_MAT3&&(o=3),r.type===e.FLOAT_MAT4&&(o=4),n[a]={type:r.type,location:e.getAttribLocation(t,a),locationSize:o}}return n}function hc(e){return e!==``}function gc(e,t){let n=t.numSpotLightShadows+t.numSpotLightMaps-t.numSpotLightShadowsWithMaps;return e.replace(/NUM_DIR_LIGHTS/g,t.numDirLights).replace(/NUM_SPOT_LIGHTS/g,t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,n).replace(/NUM_RECT_AREA_LIGHTS/g,t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,t.numPointLights).replace(/NUM_HEMI_LIGHTS/g,t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,t.numPointLightShadows)}function _c(e,t){return e.replace(/NUM_CLIPPING_PLANES/g,t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,t.numClippingPlanes-t.numClipIntersection)}var vc=/^[ \t]*#include +<([\w\d./]+)>/gm;function yc(e){return e.replace(vc,xc)}var bc=new Map;function xc(e,t){let n=oo[t];if(n===void 0){let e=bc.get(t);if(e!==void 0)n=oo[e],W(`WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.`,t,e);else throw Error(`THREE.WebGLProgram: Can not resolve #include <`+t+`>`)}return yc(n)}var Sc=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Cc(e){return e.replace(Sc,wc)}function wc(e,t,n,r){let i=``;for(let e=parseInt(t);e<parseInt(n);e++)i+=r.replace(/\[\s*i\s*\]/g,`[ `+e+` ]`).replace(/UNROLLED_LOOP_INDEX/g,e);return i}function Tc(e){let t=`precision ${e.precision} float;
	precision ${e.precision} int;
	precision ${e.precision} sampler2D;
	precision ${e.precision} samplerCube;
	precision ${e.precision} sampler3D;
	precision ${e.precision} sampler2DArray;
	precision ${e.precision} sampler2DShadow;
	precision ${e.precision} samplerCubeShadow;
	precision ${e.precision} sampler2DArrayShadow;
	precision ${e.precision} isampler2D;
	precision ${e.precision} isampler3D;
	precision ${e.precision} isamplerCube;
	precision ${e.precision} isampler2DArray;
	precision ${e.precision} usampler2D;
	precision ${e.precision} usampler3D;
	precision ${e.precision} usamplerCube;
	precision ${e.precision} usampler2DArray;
	`;return e.precision===`highp`?t+=`
#define HIGH_PRECISION`:e.precision===`mediump`?t+=`
#define MEDIUM_PRECISION`:e.precision===`lowp`&&(t+=`
#define LOW_PRECISION`),t}var Ec={1:`SHADOWMAP_TYPE_PCF`,3:`SHADOWMAP_TYPE_VSM`};function Dc(e){return Ec[e.shadowMapType]||`SHADOWMAP_TYPE_BASIC`}var Oc={301:`ENVMAP_TYPE_CUBE`,302:`ENVMAP_TYPE_CUBE`,306:`ENVMAP_TYPE_CUBE_UV`};function kc(e){return e.envMap===!1?`ENVMAP_TYPE_CUBE`:Oc[e.envMapMode]||`ENVMAP_TYPE_CUBE`}var Ac={302:`ENVMAP_MODE_REFRACTION`};function jc(e){return e.envMap===!1?`ENVMAP_MODE_REFLECTION`:Ac[e.envMapMode]||`ENVMAP_MODE_REFLECTION`}var Mc={0:`ENVMAP_BLENDING_MULTIPLY`,1:`ENVMAP_BLENDING_MIX`,2:`ENVMAP_BLENDING_ADD`};function Nc(e){return e.envMap===!1?`ENVMAP_BLENDING_NONE`:Mc[e.combine]||`ENVMAP_BLENDING_NONE`}function Pc(e){let t=e.envMapCubeUVHeight;if(t===null)return null;let n=Math.log2(t)-2,r=1/t;return{texelWidth:1/(3*Math.max(2**n,112)),texelHeight:r,maxMip:n}}function Fc(e,t,n,r){let i=e.getContext(),a=n.defines,o=n.vertexShader,s=n.fragmentShader,c=Dc(n),l=kc(n),u=jc(n),d=Nc(n),f=Pc(n),p=fc(n),m=pc(a),h=i.createProgram(),g,_,v=n.glslVersion?`#version `+n.glslVersion+`
`:``;n.isRawShaderMaterial?(g=[`#define SHADER_TYPE `+n.shaderType,`#define SHADER_NAME `+n.shaderName,m].filter(hc).join(`
`),g.length>0&&(g+=`
`),_=[`#define SHADER_TYPE `+n.shaderType,`#define SHADER_NAME `+n.shaderName,m].filter(hc).join(`
`),_.length>0&&(_+=`
`)):(g=[Tc(n),`#define SHADER_TYPE `+n.shaderType,`#define SHADER_NAME `+n.shaderName,m,n.extensionClipCullDistance?`#define USE_CLIP_DISTANCE`:``,n.batching?`#define USE_BATCHING`:``,n.batchingColor?`#define USE_BATCHING_COLOR`:``,n.instancing?`#define USE_INSTANCING`:``,n.instancingColor?`#define USE_INSTANCING_COLOR`:``,n.instancingMorph?`#define USE_INSTANCING_MORPH`:``,n.useFog&&n.fog?`#define USE_FOG`:``,n.useFog&&n.fogExp2?`#define FOG_EXP2`:``,n.map?`#define USE_MAP`:``,n.envMap?`#define USE_ENVMAP`:``,n.envMap?`#define `+u:``,n.lightMap?`#define USE_LIGHTMAP`:``,n.aoMap?`#define USE_AOMAP`:``,n.bumpMap?`#define USE_BUMPMAP`:``,n.normalMap?`#define USE_NORMALMAP`:``,n.normalMapObjectSpace?`#define USE_NORMALMAP_OBJECTSPACE`:``,n.normalMapTangentSpace?`#define USE_NORMALMAP_TANGENTSPACE`:``,n.displacementMap?`#define USE_DISPLACEMENTMAP`:``,n.emissiveMap?`#define USE_EMISSIVEMAP`:``,n.anisotropy?`#define USE_ANISOTROPY`:``,n.anisotropyMap?`#define USE_ANISOTROPYMAP`:``,n.clearcoatMap?`#define USE_CLEARCOATMAP`:``,n.clearcoatRoughnessMap?`#define USE_CLEARCOAT_ROUGHNESSMAP`:``,n.clearcoatNormalMap?`#define USE_CLEARCOAT_NORMALMAP`:``,n.iridescenceMap?`#define USE_IRIDESCENCEMAP`:``,n.iridescenceThicknessMap?`#define USE_IRIDESCENCE_THICKNESSMAP`:``,n.specularMap?`#define USE_SPECULARMAP`:``,n.specularColorMap?`#define USE_SPECULAR_COLORMAP`:``,n.specularIntensityMap?`#define USE_SPECULAR_INTENSITYMAP`:``,n.roughnessMap?`#define USE_ROUGHNESSMAP`:``,n.metalnessMap?`#define USE_METALNESSMAP`:``,n.alphaMap?`#define USE_ALPHAMAP`:``,n.alphaHash?`#define USE_ALPHAHASH`:``,n.transmission?`#define USE_TRANSMISSION`:``,n.transmissionMap?`#define USE_TRANSMISSIONMAP`:``,n.thicknessMap?`#define USE_THICKNESSMAP`:``,n.sheenColorMap?`#define USE_SHEEN_COLORMAP`:``,n.sheenRoughnessMap?`#define USE_SHEEN_ROUGHNESSMAP`:``,n.mapUv?`#define MAP_UV `+n.mapUv:``,n.alphaMapUv?`#define ALPHAMAP_UV `+n.alphaMapUv:``,n.lightMapUv?`#define LIGHTMAP_UV `+n.lightMapUv:``,n.aoMapUv?`#define AOMAP_UV `+n.aoMapUv:``,n.emissiveMapUv?`#define EMISSIVEMAP_UV `+n.emissiveMapUv:``,n.bumpMapUv?`#define BUMPMAP_UV `+n.bumpMapUv:``,n.normalMapUv?`#define NORMALMAP_UV `+n.normalMapUv:``,n.displacementMapUv?`#define DISPLACEMENTMAP_UV `+n.displacementMapUv:``,n.metalnessMapUv?`#define METALNESSMAP_UV `+n.metalnessMapUv:``,n.roughnessMapUv?`#define ROUGHNESSMAP_UV `+n.roughnessMapUv:``,n.anisotropyMapUv?`#define ANISOTROPYMAP_UV `+n.anisotropyMapUv:``,n.clearcoatMapUv?`#define CLEARCOATMAP_UV `+n.clearcoatMapUv:``,n.clearcoatNormalMapUv?`#define CLEARCOAT_NORMALMAP_UV `+n.clearcoatNormalMapUv:``,n.clearcoatRoughnessMapUv?`#define CLEARCOAT_ROUGHNESSMAP_UV `+n.clearcoatRoughnessMapUv:``,n.iridescenceMapUv?`#define IRIDESCENCEMAP_UV `+n.iridescenceMapUv:``,n.iridescenceThicknessMapUv?`#define IRIDESCENCE_THICKNESSMAP_UV `+n.iridescenceThicknessMapUv:``,n.sheenColorMapUv?`#define SHEEN_COLORMAP_UV `+n.sheenColorMapUv:``,n.sheenRoughnessMapUv?`#define SHEEN_ROUGHNESSMAP_UV `+n.sheenRoughnessMapUv:``,n.specularMapUv?`#define SPECULARMAP_UV `+n.specularMapUv:``,n.specularColorMapUv?`#define SPECULAR_COLORMAP_UV `+n.specularColorMapUv:``,n.specularIntensityMapUv?`#define SPECULAR_INTENSITYMAP_UV `+n.specularIntensityMapUv:``,n.transmissionMapUv?`#define TRANSMISSIONMAP_UV `+n.transmissionMapUv:``,n.thicknessMapUv?`#define THICKNESSMAP_UV `+n.thicknessMapUv:``,n.vertexTangents&&n.flatShading===!1?`#define USE_TANGENT`:``,n.vertexNormals?`#define HAS_NORMAL`:``,n.vertexColors?`#define USE_COLOR`:``,n.vertexAlphas?`#define USE_COLOR_ALPHA`:``,n.vertexUv1s?`#define USE_UV1`:``,n.vertexUv2s?`#define USE_UV2`:``,n.vertexUv3s?`#define USE_UV3`:``,n.pointsUvs?`#define USE_POINTS_UV`:``,n.flatShading?`#define FLAT_SHADED`:``,n.skinning?`#define USE_SKINNING`:``,n.morphTargets?`#define USE_MORPHTARGETS`:``,n.morphNormals&&n.flatShading===!1?`#define USE_MORPHNORMALS`:``,n.morphColors?`#define USE_MORPHCOLORS`:``,n.morphTargetsCount>0?`#define MORPHTARGETS_TEXTURE_STRIDE `+n.morphTextureStride:``,n.morphTargetsCount>0?`#define MORPHTARGETS_COUNT `+n.morphTargetsCount:``,n.doubleSided?`#define DOUBLE_SIDED`:``,n.flipSided?`#define FLIP_SIDED`:``,n.shadowMapEnabled?`#define USE_SHADOWMAP`:``,n.shadowMapEnabled?`#define `+c:``,n.sizeAttenuation?`#define USE_SIZEATTENUATION`:``,n.numLightProbes>0?`#define USE_LIGHT_PROBES`:``,n.logarithmicDepthBuffer?`#define USE_LOGARITHMIC_DEPTH_BUFFER`:``,n.reversedDepthBuffer?`#define USE_REVERSED_DEPTH_BUFFER`:``,`uniform mat4 modelMatrix;`,`uniform mat4 modelViewMatrix;`,`uniform mat4 projectionMatrix;`,`uniform mat4 viewMatrix;`,`uniform mat3 normalMatrix;`,`uniform vec3 cameraPosition;`,`uniform bool isOrthographic;`,`#ifdef USE_INSTANCING`,`	attribute mat4 instanceMatrix;`,`#endif`,`#ifdef USE_INSTANCING_COLOR`,`	attribute vec3 instanceColor;`,`#endif`,`#ifdef USE_INSTANCING_MORPH`,`	uniform sampler2D morphTexture;`,`#endif`,`attribute vec3 position;`,`attribute vec3 normal;`,`attribute vec2 uv;`,`#ifdef USE_UV1`,`	attribute vec2 uv1;`,`#endif`,`#ifdef USE_UV2`,`	attribute vec2 uv2;`,`#endif`,`#ifdef USE_UV3`,`	attribute vec2 uv3;`,`#endif`,`#ifdef USE_TANGENT`,`	attribute vec4 tangent;`,`#endif`,`#if defined( USE_COLOR_ALPHA )`,`	attribute vec4 color;`,`#elif defined( USE_COLOR )`,`	attribute vec3 color;`,`#endif`,`#ifdef USE_SKINNING`,`	attribute vec4 skinIndex;`,`	attribute vec4 skinWeight;`,`#endif`,`
`].filter(hc).join(`
`),_=[Tc(n),`#define SHADER_TYPE `+n.shaderType,`#define SHADER_NAME `+n.shaderName,m,n.useFog&&n.fog?`#define USE_FOG`:``,n.useFog&&n.fogExp2?`#define FOG_EXP2`:``,n.alphaToCoverage?`#define ALPHA_TO_COVERAGE`:``,n.map?`#define USE_MAP`:``,n.matcap?`#define USE_MATCAP`:``,n.envMap?`#define USE_ENVMAP`:``,n.envMap?`#define `+l:``,n.envMap?`#define `+u:``,n.envMap?`#define `+d:``,f?`#define CUBEUV_TEXEL_WIDTH `+f.texelWidth:``,f?`#define CUBEUV_TEXEL_HEIGHT `+f.texelHeight:``,f?`#define CUBEUV_MAX_MIP `+f.maxMip+`.0`:``,n.lightMap?`#define USE_LIGHTMAP`:``,n.aoMap?`#define USE_AOMAP`:``,n.bumpMap?`#define USE_BUMPMAP`:``,n.normalMap?`#define USE_NORMALMAP`:``,n.normalMapObjectSpace?`#define USE_NORMALMAP_OBJECTSPACE`:``,n.normalMapTangentSpace?`#define USE_NORMALMAP_TANGENTSPACE`:``,n.packedNormalMap?`#define USE_PACKED_NORMALMAP`:``,n.emissiveMap?`#define USE_EMISSIVEMAP`:``,n.anisotropy?`#define USE_ANISOTROPY`:``,n.anisotropyMap?`#define USE_ANISOTROPYMAP`:``,n.clearcoat?`#define USE_CLEARCOAT`:``,n.clearcoatMap?`#define USE_CLEARCOATMAP`:``,n.clearcoatRoughnessMap?`#define USE_CLEARCOAT_ROUGHNESSMAP`:``,n.clearcoatNormalMap?`#define USE_CLEARCOAT_NORMALMAP`:``,n.dispersion?`#define USE_DISPERSION`:``,n.iridescence?`#define USE_IRIDESCENCE`:``,n.iridescenceMap?`#define USE_IRIDESCENCEMAP`:``,n.iridescenceThicknessMap?`#define USE_IRIDESCENCE_THICKNESSMAP`:``,n.specularMap?`#define USE_SPECULARMAP`:``,n.specularColorMap?`#define USE_SPECULAR_COLORMAP`:``,n.specularIntensityMap?`#define USE_SPECULAR_INTENSITYMAP`:``,n.roughnessMap?`#define USE_ROUGHNESSMAP`:``,n.metalnessMap?`#define USE_METALNESSMAP`:``,n.alphaMap?`#define USE_ALPHAMAP`:``,n.alphaTest?`#define USE_ALPHATEST`:``,n.alphaHash?`#define USE_ALPHAHASH`:``,n.sheen?`#define USE_SHEEN`:``,n.sheenColorMap?`#define USE_SHEEN_COLORMAP`:``,n.sheenRoughnessMap?`#define USE_SHEEN_ROUGHNESSMAP`:``,n.transmission?`#define USE_TRANSMISSION`:``,n.transmissionMap?`#define USE_TRANSMISSIONMAP`:``,n.thicknessMap?`#define USE_THICKNESSMAP`:``,n.vertexTangents&&n.flatShading===!1?`#define USE_TANGENT`:``,n.vertexColors||n.instancingColor?`#define USE_COLOR`:``,n.vertexAlphas||n.batchingColor?`#define USE_COLOR_ALPHA`:``,n.vertexUv1s?`#define USE_UV1`:``,n.vertexUv2s?`#define USE_UV2`:``,n.vertexUv3s?`#define USE_UV3`:``,n.pointsUvs?`#define USE_POINTS_UV`:``,n.gradientMap?`#define USE_GRADIENTMAP`:``,n.flatShading?`#define FLAT_SHADED`:``,n.doubleSided?`#define DOUBLE_SIDED`:``,n.flipSided?`#define FLIP_SIDED`:``,n.shadowMapEnabled?`#define USE_SHADOWMAP`:``,n.shadowMapEnabled?`#define `+c:``,n.premultipliedAlpha?`#define PREMULTIPLIED_ALPHA`:``,n.numLightProbes>0?`#define USE_LIGHT_PROBES`:``,n.numLightProbeGrids>0?`#define USE_LIGHT_PROBES_GRID`:``,n.decodeVideoTexture?`#define DECODE_VIDEO_TEXTURE`:``,n.decodeVideoTextureEmissive?`#define DECODE_VIDEO_TEXTURE_EMISSIVE`:``,n.logarithmicDepthBuffer?`#define USE_LOGARITHMIC_DEPTH_BUFFER`:``,n.reversedDepthBuffer?`#define USE_REVERSED_DEPTH_BUFFER`:``,`uniform mat4 viewMatrix;`,`uniform vec3 cameraPosition;`,`uniform bool isOrthographic;`,n.toneMapping===0?``:`#define TONE_MAPPING`,n.toneMapping===0?``:oo.tonemapping_pars_fragment,n.toneMapping===0?``:lc(`toneMapping`,n.toneMapping),n.dithering?`#define DITHERING`:``,n.opaque?`#define OPAQUE`:``,oo.colorspace_pars_fragment,sc(`linearToOutputTexel`,n.outputColorSpace),dc(),n.useDepthPacking?`#define DEPTH_PACKING `+n.depthPacking:``,`
`].filter(hc).join(`
`)),o=yc(o),o=gc(o,n),o=_c(o,n),s=yc(s),s=gc(s,n),s=_c(s,n),o=Cc(o),s=Cc(s),n.isRawShaderMaterial!==!0&&(v=`#version 300 es
`,g=[p,`#define attribute in`,`#define varying out`,`#define texture2D texture`].join(`
`)+`
`+g,_=[`#define varying in`,n.glslVersion===`300 es`?``:`layout(location = 0) out highp vec4 pc_fragColor;`,n.glslVersion===`300 es`?``:`#define gl_FragColor pc_fragColor`,`#define gl_FragDepthEXT gl_FragDepth`,`#define texture2D texture`,`#define textureCube texture`,`#define texture2DProj textureProj`,`#define texture2DLodEXT textureLod`,`#define texture2DProjLodEXT textureProjLod`,`#define textureCubeLodEXT textureLod`,`#define texture2DGradEXT textureGrad`,`#define texture2DProjGradEXT textureProjGrad`,`#define textureCubeGradEXT textureGrad`].join(`
`)+`
`+_);let y=v+g+o,b=v+_+s,x=ec(i,i.VERTEX_SHADER,y),S=ec(i,i.FRAGMENT_SHADER,b);i.attachShader(h,x),i.attachShader(h,S),n.index0AttributeName===void 0?n.hasPositionAttribute===!0&&i.bindAttribLocation(h,0,`position`):i.bindAttribLocation(h,0,n.index0AttributeName),i.linkProgram(h);function C(t){if(e.debug.checkShaderErrors){let n=i.getProgramInfoLog(h)||``,r=i.getShaderInfoLog(x)||``,a=i.getShaderInfoLog(S)||``,o=n.trim(),s=r.trim(),c=a.trim(),l=!0,u=!0;if(i.getProgramParameter(h,i.LINK_STATUS)===!1)if(l=!1,typeof e.debug.onShaderError==`function`)e.debug.onShaderError(i,h,x,S);else{let e=oc(i,x,`vertex`),n=oc(i,S,`fragment`);G(`WebGLProgram: Shader Error `+i.getError()+` - VALIDATE_STATUS `+i.getProgramParameter(h,i.VALIDATE_STATUS)+`

Material Name: `+t.name+`
Material Type: `+t.type+`

Program Info Log: `+o+`
`+e+`
`+n)}else o===``?(s===``||c===``)&&(u=!1):W(`WebGLProgram: Program Info Log:`,o);u&&(t.diagnostics={runnable:l,programLog:o,vertexShader:{log:s,prefix:g},fragmentShader:{log:c,prefix:_}})}i.deleteShader(x),i.deleteShader(S),w=new $s(i,h),T=mc(i,h)}let w;this.getUniforms=function(){return w===void 0&&C(this),w};let T;this.getAttributes=function(){return T===void 0&&C(this),T};let E=n.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return E===!1&&(E=i.getProgramParameter(h,tc)),E},this.destroy=function(){r.releaseStatesOfProgram(this),i.deleteProgram(h),this.program=void 0},this.type=n.shaderType,this.name=n.shaderName,this.id=nc++,this.cacheKey=t,this.usedTimes=1,this.program=h,this.vertexShader=x,this.fragmentShader=S,this}var Ic=0,Lc=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,t,n){let r=this._getShaderCacheForMaterial(e);return r.has(t)===!1&&(r.add(t),t.usedTimes++),r.has(n)===!1&&(r.add(n),n.usedTimes++),this}remove(e){let t=this.materialCache.get(e);for(let e of t)e.usedTimes--,e.usedTimes===0&&this.shaderCache.delete(e.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let t=this.materialCache,n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){let t=this.shaderCache,n=t.get(e);return n===void 0&&(n=new Rc(e),t.set(e,n)),n}},Rc=class{constructor(e){this.id=Ic++,this.code=e,this.usedTimes=0}};function zc(e){return e===1030||e===37490||e===36285}function Bc(e,t,n,r,i,a){let o=new pn,s=new Lc,c=new Set,l=[],u=new Map,d=r.logarithmicDepthBuffer,f=r.precision,p={MeshDepthMaterial:`depth`,MeshDistanceMaterial:`distance`,MeshNormalMaterial:`normal`,MeshBasicMaterial:`basic`,MeshLambertMaterial:`lambert`,MeshPhongMaterial:`phong`,MeshToonMaterial:`toon`,MeshStandardMaterial:`physical`,MeshPhysicalMaterial:`physical`,MeshMatcapMaterial:`matcap`,LineBasicMaterial:`basic`,LineDashedMaterial:`dashed`,PointsMaterial:`points`,ShadowMaterial:`shadow`,SpriteMaterial:`sprite`};function m(e){return c.add(e),e===0?`uv`:`uv${e}`}function h(i,o,l,u,h,g){let _=u.fog,v=h.geometry,y=i.isMeshStandardMaterial||i.isMeshLambertMaterial||i.isMeshPhongMaterial?u.environment:null,b=i.isMeshStandardMaterial||i.isMeshLambertMaterial&&!i.envMap||i.isMeshPhongMaterial&&!i.envMap,x=t.get(i.envMap||y,b),S=x&&x.mapping===306?x.image.height:null,C=p[i.type];i.precision!==null&&(f=r.getMaxPrecision(i.precision),f!==i.precision&&W(`WebGLProgram.getParameters:`,i.precision,`not supported, using`,f,`instead.`));let w=v.morphAttributes.position||v.morphAttributes.normal||v.morphAttributes.color,T=w===void 0?0:w.length,E=0;v.morphAttributes.position!==void 0&&(E=1),v.morphAttributes.normal!==void 0&&(E=2),v.morphAttributes.color!==void 0&&(E=3);let D,O,k,A;if(C){let e=so[C];D=e.vertexShader,O=e.fragmentShader}else{D=i.vertexShader,O=i.fragmentShader;let e=s.getVertexShaderStage(i),t=s.getFragmentShaderStage(i);s.update(i,e,t),k=e.id,A=t.id}let j=e.getRenderTarget(),M=e.state.buffers.depth.getReversed(),N=h.isInstancedMesh===!0,P=h.isBatchedMesh===!0,F=!!i.map,I=!!i.matcap,ee=!!x,te=!!i.aoMap,L=!!i.lightMap,R=!!i.bumpMap&&i.wireframe===!1,ne=!!i.normalMap,re=!!i.displacementMap,ie=!!i.emissiveMap,z=!!i.metalnessMap,ae=!!i.roughnessMap,oe=i.anisotropy>0,se=i.clearcoat>0,ce=i.dispersion>0,le=i.iridescence>0,B=i.sheen>0,ue=i.transmission>0,V=oe&&!!i.anisotropyMap,de=se&&!!i.clearcoatMap,fe=se&&!!i.clearcoatNormalMap,pe=se&&!!i.clearcoatRoughnessMap,me=le&&!!i.iridescenceMap,he=le&&!!i.iridescenceThicknessMap,ge=B&&!!i.sheenColorMap,_e=B&&!!i.sheenRoughnessMap,ve=!!i.specularMap,ye=!!i.specularColorMap,be=!!i.specularIntensityMap,xe=ue&&!!i.transmissionMap,Se=ue&&!!i.thicknessMap,Ce=!!i.gradientMap,we=!!i.alphaMap,Te=i.alphaTest>0,Ee=!!i.alphaHash,H=!!i.extensions,De=0;i.toneMapped&&(j===null||j.isXRRenderTarget===!0)&&(De=e.toneMapping);let Oe={shaderID:C,shaderType:i.type,shaderName:i.name,vertexShader:D,fragmentShader:O,defines:i.defines,customVertexShaderID:k,customFragmentShaderID:A,isRawShaderMaterial:i.isRawShaderMaterial===!0,glslVersion:i.glslVersion,precision:f,batching:P,batchingColor:P&&h._colorsTexture!==null,instancing:N,instancingColor:N&&h.instanceColor!==null,instancingMorph:N&&h.morphTexture!==null,outputColorSpace:j===null?e.outputColorSpace:j.isXRRenderTarget===!0?j.texture.colorSpace:Y.workingColorSpace,alphaToCoverage:!!i.alphaToCoverage,map:F,matcap:I,envMap:ee,envMapMode:ee&&x.mapping,envMapCubeUVHeight:S,aoMap:te,lightMap:L,bumpMap:R,normalMap:ne,displacementMap:re,emissiveMap:ie,normalMapObjectSpace:ne&&i.normalMapType===1,normalMapTangentSpace:ne&&i.normalMapType===0,packedNormalMap:ne&&i.normalMapType===0&&zc(i.normalMap.format),metalnessMap:z,roughnessMap:ae,anisotropy:oe,anisotropyMap:V,clearcoat:se,clearcoatMap:de,clearcoatNormalMap:fe,clearcoatRoughnessMap:pe,dispersion:ce,iridescence:le,iridescenceMap:me,iridescenceThicknessMap:he,sheen:B,sheenColorMap:ge,sheenRoughnessMap:_e,specularMap:ve,specularColorMap:ye,specularIntensityMap:be,transmission:ue,transmissionMap:xe,thicknessMap:Se,gradientMap:Ce,opaque:i.transparent===!1&&i.blending===1&&i.alphaToCoverage===!1,alphaMap:we,alphaTest:Te,alphaHash:Ee,combine:i.combine,mapUv:F&&m(i.map.channel),aoMapUv:te&&m(i.aoMap.channel),lightMapUv:L&&m(i.lightMap.channel),bumpMapUv:R&&m(i.bumpMap.channel),normalMapUv:ne&&m(i.normalMap.channel),displacementMapUv:re&&m(i.displacementMap.channel),emissiveMapUv:ie&&m(i.emissiveMap.channel),metalnessMapUv:z&&m(i.metalnessMap.channel),roughnessMapUv:ae&&m(i.roughnessMap.channel),anisotropyMapUv:V&&m(i.anisotropyMap.channel),clearcoatMapUv:de&&m(i.clearcoatMap.channel),clearcoatNormalMapUv:fe&&m(i.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:pe&&m(i.clearcoatRoughnessMap.channel),iridescenceMapUv:me&&m(i.iridescenceMap.channel),iridescenceThicknessMapUv:he&&m(i.iridescenceThicknessMap.channel),sheenColorMapUv:ge&&m(i.sheenColorMap.channel),sheenRoughnessMapUv:_e&&m(i.sheenRoughnessMap.channel),specularMapUv:ve&&m(i.specularMap.channel),specularColorMapUv:ye&&m(i.specularColorMap.channel),specularIntensityMapUv:be&&m(i.specularIntensityMap.channel),transmissionMapUv:xe&&m(i.transmissionMap.channel),thicknessMapUv:Se&&m(i.thicknessMap.channel),alphaMapUv:we&&m(i.alphaMap.channel),vertexTangents:!!v.attributes.tangent&&(ne||oe),vertexNormals:!!v.attributes.normal,vertexColors:i.vertexColors,vertexAlphas:i.vertexColors===!0&&!!v.attributes.color&&v.attributes.color.itemSize===4,pointsUvs:h.isPoints===!0&&!!v.attributes.uv&&(F||we),fog:!!_,useFog:i.fog===!0,fogExp2:!!_&&_.isFogExp2,flatShading:i.wireframe===!1&&(i.flatShading===!0||v.attributes.normal===void 0&&ne===!1&&(i.isMeshLambertMaterial||i.isMeshPhongMaterial||i.isMeshStandardMaterial||i.isMeshPhysicalMaterial)),sizeAttenuation:i.sizeAttenuation===!0,logarithmicDepthBuffer:d,reversedDepthBuffer:M,skinning:h.isSkinnedMesh===!0,hasPositionAttribute:v.attributes.position!==void 0,morphTargets:v.morphAttributes.position!==void 0,morphNormals:v.morphAttributes.normal!==void 0,morphColors:v.morphAttributes.color!==void 0,morphTargetsCount:T,morphTextureStride:E,numDirLights:o.directional.length,numPointLights:o.point.length,numSpotLights:o.spot.length,numSpotLightMaps:o.spotLightMap.length,numRectAreaLights:o.rectArea.length,numHemiLights:o.hemi.length,numDirLightShadows:o.directionalShadowMap.length,numPointLightShadows:o.pointShadowMap.length,numSpotLightShadows:o.spotShadowMap.length,numSpotLightShadowsWithMaps:o.numSpotLightShadowsWithMaps,numLightProbes:o.numLightProbes,numLightProbeGrids:g.length,numClippingPlanes:a.numPlanes,numClipIntersection:a.numIntersection,dithering:i.dithering,shadowMapEnabled:e.shadowMap.enabled&&l.length>0,shadowMapType:e.shadowMap.type,toneMapping:De,decodeVideoTexture:F&&i.map.isVideoTexture===!0&&Y.getTransfer(i.map.colorSpace)===`srgb`,decodeVideoTextureEmissive:ie&&i.emissiveMap.isVideoTexture===!0&&Y.getTransfer(i.emissiveMap.colorSpace)===`srgb`,premultipliedAlpha:i.premultipliedAlpha,doubleSided:i.side===2,flipSided:i.side===1,useDepthPacking:i.depthPacking>=0,depthPacking:i.depthPacking||0,index0AttributeName:i.index0AttributeName,extensionClipCullDistance:H&&i.extensions.clipCullDistance===!0&&n.has(`WEBGL_clip_cull_distance`),extensionMultiDraw:(H&&i.extensions.multiDraw===!0||P)&&n.has(`WEBGL_multi_draw`),rendererExtensionParallelShaderCompile:n.has(`KHR_parallel_shader_compile`),customProgramCacheKey:i.customProgramCacheKey()};return Oe.vertexUv1s=c.has(1),Oe.vertexUv2s=c.has(2),Oe.vertexUv3s=c.has(3),c.clear(),Oe}function g(t){let n=[];if(t.shaderID?n.push(t.shaderID):(n.push(t.customVertexShaderID),n.push(t.customFragmentShaderID)),t.defines!==void 0)for(let e in t.defines)n.push(e),n.push(t.defines[e]);return t.isRawShaderMaterial===!1&&(_(n,t),v(n,t),n.push(e.outputColorSpace)),n.push(t.customProgramCacheKey),n.join()}function _(e,t){e.push(t.precision),e.push(t.outputColorSpace),e.push(t.envMapMode),e.push(t.envMapCubeUVHeight),e.push(t.mapUv),e.push(t.alphaMapUv),e.push(t.lightMapUv),e.push(t.aoMapUv),e.push(t.bumpMapUv),e.push(t.normalMapUv),e.push(t.displacementMapUv),e.push(t.emissiveMapUv),e.push(t.metalnessMapUv),e.push(t.roughnessMapUv),e.push(t.anisotropyMapUv),e.push(t.clearcoatMapUv),e.push(t.clearcoatNormalMapUv),e.push(t.clearcoatRoughnessMapUv),e.push(t.iridescenceMapUv),e.push(t.iridescenceThicknessMapUv),e.push(t.sheenColorMapUv),e.push(t.sheenRoughnessMapUv),e.push(t.specularMapUv),e.push(t.specularColorMapUv),e.push(t.specularIntensityMapUv),e.push(t.transmissionMapUv),e.push(t.thicknessMapUv),e.push(t.combine),e.push(t.fogExp2),e.push(t.sizeAttenuation),e.push(t.morphTargetsCount),e.push(t.morphAttributeCount),e.push(t.numDirLights),e.push(t.numPointLights),e.push(t.numSpotLights),e.push(t.numSpotLightMaps),e.push(t.numHemiLights),e.push(t.numRectAreaLights),e.push(t.numDirLightShadows),e.push(t.numPointLightShadows),e.push(t.numSpotLightShadows),e.push(t.numSpotLightShadowsWithMaps),e.push(t.numLightProbes),e.push(t.shadowMapType),e.push(t.toneMapping),e.push(t.numClippingPlanes),e.push(t.numClipIntersection),e.push(t.depthPacking)}function v(e,t){o.disableAll(),t.instancing&&o.enable(0),t.instancingColor&&o.enable(1),t.instancingMorph&&o.enable(2),t.matcap&&o.enable(3),t.envMap&&o.enable(4),t.normalMapObjectSpace&&o.enable(5),t.normalMapTangentSpace&&o.enable(6),t.clearcoat&&o.enable(7),t.iridescence&&o.enable(8),t.alphaTest&&o.enable(9),t.vertexColors&&o.enable(10),t.vertexAlphas&&o.enable(11),t.vertexUv1s&&o.enable(12),t.vertexUv2s&&o.enable(13),t.vertexUv3s&&o.enable(14),t.vertexTangents&&o.enable(15),t.anisotropy&&o.enable(16),t.alphaHash&&o.enable(17),t.batching&&o.enable(18),t.dispersion&&o.enable(19),t.batchingColor&&o.enable(20),t.gradientMap&&o.enable(21),t.packedNormalMap&&o.enable(22),t.vertexNormals&&o.enable(23),e.push(o.mask),o.disableAll(),t.fog&&o.enable(0),t.useFog&&o.enable(1),t.flatShading&&o.enable(2),t.logarithmicDepthBuffer&&o.enable(3),t.reversedDepthBuffer&&o.enable(4),t.skinning&&o.enable(5),t.morphTargets&&o.enable(6),t.morphNormals&&o.enable(7),t.morphColors&&o.enable(8),t.premultipliedAlpha&&o.enable(9),t.shadowMapEnabled&&o.enable(10),t.doubleSided&&o.enable(11),t.flipSided&&o.enable(12),t.useDepthPacking&&o.enable(13),t.dithering&&o.enable(14),t.transmission&&o.enable(15),t.sheen&&o.enable(16),t.opaque&&o.enable(17),t.pointsUvs&&o.enable(18),t.decodeVideoTexture&&o.enable(19),t.decodeVideoTextureEmissive&&o.enable(20),t.alphaToCoverage&&o.enable(21),t.numLightProbeGrids>0&&o.enable(22),t.hasPositionAttribute&&o.enable(23),e.push(o.mask)}function y(e){let t=p[e.type],n;if(t){let e=so[t];n=$i.clone(e.uniforms)}else n=e.uniforms;return n}function b(t,n){let r=u.get(n);return r===void 0?(r=new Fc(e,n,t,i),l.push(r),u.set(n,r)):++r.usedTimes,r}function x(e){if(--e.usedTimes===0){let t=l.indexOf(e);l[t]=l[l.length-1],l.pop(),u.delete(e.cacheKey),e.destroy()}}function S(e){s.remove(e)}function C(){s.dispose()}return{getParameters:h,getProgramCacheKey:g,getUniforms:y,acquireProgram:b,releaseProgram:x,releaseShaderCache:S,programs:l,dispose:C}}function Vc(){let e=new WeakMap;function t(t){return e.has(t)}function n(t){let n=e.get(t);return n===void 0&&(n={},e.set(t,n)),n}function r(t){e.delete(t)}function i(t,n,r){e.get(t)[n]=r}function a(){e=new WeakMap}return{has:t,get:n,remove:r,update:i,dispose:a}}function Hc(e,t){return e.groupOrder===t.groupOrder?e.renderOrder===t.renderOrder?e.material.id===t.material.id?e.materialVariant===t.materialVariant?e.z===t.z?e.id-t.id:e.z-t.z:e.materialVariant-t.materialVariant:e.material.id-t.material.id:e.renderOrder-t.renderOrder:e.groupOrder-t.groupOrder}function Uc(e,t){return e.groupOrder===t.groupOrder?e.renderOrder===t.renderOrder?e.z===t.z?e.id-t.id:t.z-e.z:e.renderOrder-t.renderOrder:e.groupOrder-t.groupOrder}function Wc(){let e=[],t=0,n=[],r=[],i=[];function a(){t=0,n.length=0,r.length=0,i.length=0}function o(e){let t=0;return e.isInstancedMesh&&(t+=2),e.isSkinnedMesh&&(t+=1),t}function s(n,r,i,a,s,c){let l=e[t];return l===void 0?(l={id:n.id,object:n,geometry:r,material:i,materialVariant:o(n),groupOrder:a,renderOrder:n.renderOrder,z:s,group:c},e[t]=l):(l.id=n.id,l.object=n,l.geometry=r,l.material=i,l.materialVariant=o(n),l.groupOrder=a,l.renderOrder=n.renderOrder,l.z=s,l.group=c),t++,l}function c(e,t,a,o,c,l){let u=s(e,t,a,o,c,l);a.transmission>0?r.push(u):a.transparent===!0?i.push(u):n.push(u)}function l(e,t,a,o,c,l){let u=s(e,t,a,o,c,l);a.transmission>0?r.unshift(u):a.transparent===!0?i.unshift(u):n.unshift(u)}function u(e,t,a){n.length>1&&n.sort(e||Hc),r.length>1&&r.sort(t||Uc),i.length>1&&i.sort(t||Uc),a&&(n.reverse(),r.reverse(),i.reverse())}function d(){for(let n=t,r=e.length;n<r;n++){let t=e[n];if(t.id===null)break;t.id=null,t.object=null,t.geometry=null,t.material=null,t.group=null}}return{opaque:n,transmissive:r,transparent:i,init:a,push:c,unshift:l,finish:d,sort:u}}function Gc(){let e=new WeakMap;function t(t,n){let r=e.get(t),i;return r===void 0?(i=new Wc,e.set(t,[i])):n>=r.length?(i=new Wc,r.push(i)):i=r[n],i}function n(){e=new WeakMap}return{get:t,dispose:n}}function Kc(){let e={};return{get:function(t){if(e[t.id]!==void 0)return e[t.id];let n;switch(t.type){case`DirectionalLight`:n={direction:new q,color:new Ln};break;case`SpotLight`:n={position:new q,direction:new q,color:new Ln,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case`PointLight`:n={position:new q,color:new Ln,distance:0,decay:0};break;case`HemisphereLight`:n={direction:new q,skyColor:new Ln,groundColor:new Ln};break;case`RectAreaLight`:n={color:new Ln,position:new q,halfWidth:new q,halfHeight:new q};break}return e[t.id]=n,n}}}function qc(){let e={};return{get:function(t){if(e[t.id]!==void 0)return e[t.id];let n;switch(t.type){case`DirectionalLight`:n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Mt};break;case`SpotLight`:n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Mt};break;case`PointLight`:n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Mt,shadowCameraNear:1,shadowCameraFar:1e3};break}return e[t.id]=n,n}}}var Jc=0;function Yc(e,t){return(t.castShadow?2:0)-(e.castShadow?2:0)+(t.map?1:0)-(e.map?1:0)}function Xc(e){let t=new Kc,n=qc(),r={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let e=0;e<9;e++)r.probe.push(new q);let i=new q,a=new tn,o=new tn;function s(i){let a=0,o=0,s=0;for(let e=0;e<9;e++)r.probe[e].set(0,0,0);let c=0,l=0,u=0,d=0,f=0,p=0,m=0,h=0,g=0,_=0,v=0;i.sort(Yc);for(let e=0,y=i.length;e<y;e++){let y=i[e],b=y.color,x=y.intensity,S=y.distance,C=null;if(y.shadow&&y.shadow.map&&(C=y.shadow.map.texture.format===1030?y.shadow.map.texture:y.shadow.map.depthTexture||y.shadow.map.texture),y.isAmbientLight)a+=b.r*x,o+=b.g*x,s+=b.b*x;else if(y.isLightProbe){for(let e=0;e<9;e++)r.probe[e].addScaledVector(y.sh.coefficients[e],x);v++}else if(y.isDirectionalLight){let e=t.get(y);if(e.color.copy(y.color).multiplyScalar(y.intensity),y.castShadow){let e=y.shadow,t=n.get(y);t.shadowIntensity=e.intensity,t.shadowBias=e.bias,t.shadowNormalBias=e.normalBias,t.shadowRadius=e.radius,t.shadowMapSize=e.mapSize,r.directionalShadow[c]=t,r.directionalShadowMap[c]=C,r.directionalShadowMatrix[c]=y.shadow.matrix,p++}r.directional[c]=e,c++}else if(y.isSpotLight){let e=t.get(y);e.position.setFromMatrixPosition(y.matrixWorld),e.color.copy(b).multiplyScalar(x),e.distance=S,e.coneCos=Math.cos(y.angle),e.penumbraCos=Math.cos(y.angle*(1-y.penumbra)),e.decay=y.decay,r.spot[u]=e;let i=y.shadow;if(y.map&&(r.spotLightMap[g]=y.map,g++,i.updateMatrices(y),y.castShadow&&_++),r.spotLightMatrix[u]=i.matrix,y.castShadow){let e=n.get(y);e.shadowIntensity=i.intensity,e.shadowBias=i.bias,e.shadowNormalBias=i.normalBias,e.shadowRadius=i.radius,e.shadowMapSize=i.mapSize,r.spotShadow[u]=e,r.spotShadowMap[u]=C,h++}u++}else if(y.isRectAreaLight){let e=t.get(y);e.color.copy(b).multiplyScalar(x),e.halfWidth.set(y.width*.5,0,0),e.halfHeight.set(0,y.height*.5,0),r.rectArea[d]=e,d++}else if(y.isPointLight){let e=t.get(y);if(e.color.copy(y.color).multiplyScalar(y.intensity),e.distance=y.distance,e.decay=y.decay,y.castShadow){let e=y.shadow,t=n.get(y);t.shadowIntensity=e.intensity,t.shadowBias=e.bias,t.shadowNormalBias=e.normalBias,t.shadowRadius=e.radius,t.shadowMapSize=e.mapSize,t.shadowCameraNear=e.camera.near,t.shadowCameraFar=e.camera.far,r.pointShadow[l]=t,r.pointShadowMap[l]=C,r.pointShadowMatrix[l]=y.shadow.matrix,m++}r.point[l]=e,l++}else if(y.isHemisphereLight){let e=t.get(y);e.skyColor.copy(y.color).multiplyScalar(x),e.groundColor.copy(y.groundColor).multiplyScalar(x),r.hemi[f]=e,f++}}d>0&&(e.has(`OES_texture_float_linear`)===!0?(r.rectAreaLTC1=X.LTC_FLOAT_1,r.rectAreaLTC2=X.LTC_FLOAT_2):(r.rectAreaLTC1=X.LTC_HALF_1,r.rectAreaLTC2=X.LTC_HALF_2)),r.ambient[0]=a,r.ambient[1]=o,r.ambient[2]=s;let y=r.hash;(y.directionalLength!==c||y.pointLength!==l||y.spotLength!==u||y.rectAreaLength!==d||y.hemiLength!==f||y.numDirectionalShadows!==p||y.numPointShadows!==m||y.numSpotShadows!==h||y.numSpotMaps!==g||y.numLightProbes!==v)&&(r.directional.length=c,r.spot.length=u,r.rectArea.length=d,r.point.length=l,r.hemi.length=f,r.directionalShadow.length=p,r.directionalShadowMap.length=p,r.pointShadow.length=m,r.pointShadowMap.length=m,r.spotShadow.length=h,r.spotShadowMap.length=h,r.directionalShadowMatrix.length=p,r.pointShadowMatrix.length=m,r.spotLightMatrix.length=h+g-_,r.spotLightMap.length=g,r.numSpotLightShadowsWithMaps=_,r.numLightProbes=v,y.directionalLength=c,y.pointLength=l,y.spotLength=u,y.rectAreaLength=d,y.hemiLength=f,y.numDirectionalShadows=p,y.numPointShadows=m,y.numSpotShadows=h,y.numSpotMaps=g,y.numLightProbes=v,r.version=Jc++)}function c(e,t){let n=0,s=0,c=0,l=0,u=0,d=t.matrixWorldInverse;for(let t=0,f=e.length;t<f;t++){let f=e[t];if(f.isDirectionalLight){let e=r.directional[n];e.direction.setFromMatrixPosition(f.matrixWorld),i.setFromMatrixPosition(f.target.matrixWorld),e.direction.sub(i),e.direction.transformDirection(d),n++}else if(f.isSpotLight){let e=r.spot[c];e.position.setFromMatrixPosition(f.matrixWorld),e.position.applyMatrix4(d),e.direction.setFromMatrixPosition(f.matrixWorld),i.setFromMatrixPosition(f.target.matrixWorld),e.direction.sub(i),e.direction.transformDirection(d),c++}else if(f.isRectAreaLight){let e=r.rectArea[l];e.position.setFromMatrixPosition(f.matrixWorld),e.position.applyMatrix4(d),o.identity(),a.copy(f.matrixWorld),a.premultiply(d),o.extractRotation(a),e.halfWidth.set(f.width*.5,0,0),e.halfHeight.set(0,f.height*.5,0),e.halfWidth.applyMatrix4(o),e.halfHeight.applyMatrix4(o),l++}else if(f.isPointLight){let e=r.point[s];e.position.setFromMatrixPosition(f.matrixWorld),e.position.applyMatrix4(d),s++}else if(f.isHemisphereLight){let e=r.hemi[u];e.direction.setFromMatrixPosition(f.matrixWorld),e.direction.transformDirection(d),u++}}}return{setup:s,setupView:c,state:r}}function Zc(e){let t=new Xc(e),n=[],r=[],i=[];function a(e){d.camera=e,n.length=0,r.length=0,i.length=0}function o(e){n.push(e)}function s(e){r.push(e)}function c(e){i.push(e)}function l(){t.setup(n)}function u(e){t.setupView(n,e)}let d={lightsArray:n,shadowsArray:r,lightProbeGridArray:i,camera:null,lights:t,transmissionRenderTarget:{},textureUnits:0};return{init:a,state:d,setupLights:l,setupLightsView:u,pushLight:o,pushShadow:s,pushLightProbeGrid:c}}function Qc(e){let t=new WeakMap;function n(n,r=0){let i=t.get(n),a;return i===void 0?(a=new Zc(e),t.set(n,[a])):r>=i.length?(a=new Zc(e),i.push(a)):a=i[r],a}function r(){t=new WeakMap}return{get:n,dispose:r}}var $c=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,el=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,tl=[new q(1,0,0),new q(-1,0,0),new q(0,1,0),new q(0,-1,0),new q(0,0,1),new q(0,0,-1)],nl=[new q(0,-1,0),new q(0,-1,0),new q(0,0,1),new q(0,0,-1),new q(0,-1,0),new q(0,-1,0)],rl=new tn,il=new q,al=new q;function ol(e,t,n){let r=new Ni,i=new Mt,a=new Mt,o=new Xt,s=new aa,c=new oa,l={},u=n.maxTextureSize,d={0:1,1:0,2:2},f=new na({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Mt},radius:{value:4}},vertexShader:$c,fragmentShader:el}),p=f.clone();p.defines.HORIZONTAL_PASS=1;let m=new Pr;m.setAttribute(`position`,new yr(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let h=new Si(m,f),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=1;let _=this.type;this.render=function(t,n,s){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||t.length===0)return;this.type===2&&(W(`WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead.`),this.type=1);let c=e.getRenderTarget(),l=e.getActiveCubeFace(),d=e.getActiveMipmapLevel(),f=e.state;f.setBlending(0),f.buffers.depth.getReversed()===!0?f.buffers.color.setClear(0,0,0,0):f.buffers.color.setClear(1,1,1,1),f.buffers.depth.setTest(!0),f.setScissorTest(!1);let p=_!==this.type;p&&n.traverse(function(e){e.material&&(Array.isArray(e.material)?e.material.forEach(e=>e.needsUpdate=!0):e.material.needsUpdate=!0)});for(let c=0,l=t.length;c<l;c++){let l=t[c],d=l.shadow;if(d===void 0){W(`WebGLShadowMap:`,l,`has no shadow.`);continue}if(d.autoUpdate===!1&&d.needsUpdate===!1)continue;i.copy(d.mapSize);let m=d.getFrameExtents();i.multiply(m),a.copy(d.mapSize),(i.x>u||i.y>u)&&(i.x>u&&(a.x=Math.floor(u/m.x),i.x=a.x*m.x,d.mapSize.x=a.x),i.y>u&&(a.y=Math.floor(u/m.y),i.y=a.y*m.y,d.mapSize.y=a.y));let h=e.state.buffers.depth.getReversed();if(d.camera._reversedDepth=h,d.map===null||p===!0){if(d.map!==null&&(d.map.depthTexture!==null&&(d.map.depthTexture.dispose(),d.map.depthTexture=null),d.map.dispose()),this.type===3){if(l.isPointLight){W(`WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.`);continue}d.map=new Qt(i.x,i.y,{format:me,type:ie,minFilter:N,magFilter:N,generateMipmaps:!1}),d.map.texture.name=l.name+`.shadowMap`,d.map.depthTexture=new Ui(i.x,i.y,re),d.map.depthTexture.name=l.name+`.shadowMapDepth`,d.map.depthTexture.format=V,d.map.depthTexture.compareFunction=null,d.map.depthTexture.minFilter=A,d.map.depthTexture.magFilter=A}else l.isPointLight?(d.map=new Lo(i.x),d.map.depthTexture=new Wi(i.x,ne)):(d.map=new Qt(i.x,i.y),d.map.depthTexture=new Ui(i.x,i.y,ne)),d.map.depthTexture.name=l.name+`.shadowMap`,d.map.depthTexture.format=V,this.type===1?(d.map.depthTexture.compareFunction=h?518:515,d.map.depthTexture.minFilter=N,d.map.depthTexture.magFilter=N):(d.map.depthTexture.compareFunction=null,d.map.depthTexture.minFilter=A,d.map.depthTexture.magFilter=A);d.camera.updateProjectionMatrix()}let g=d.map.isWebGLCubeRenderTarget?6:1;for(let t=0;t<g;t++){if(d.map.isWebGLCubeRenderTarget)e.setRenderTarget(d.map,t),e.clear();else{t===0&&(e.setRenderTarget(d.map),e.clear());let n=d.getViewport(t);o.set(a.x*n.x,a.y*n.y,a.x*n.z,a.y*n.w),f.viewport(o)}if(l.isPointLight){let e=d.camera,n=d.matrix,r=l.distance||e.far;r!==e.far&&(e.far=r,e.updateProjectionMatrix()),il.setFromMatrixPosition(l.matrixWorld),e.position.copy(il),al.copy(e.position),al.add(tl[t]),e.up.copy(nl[t]),e.lookAt(al),e.updateMatrixWorld(),n.makeTranslation(-il.x,-il.y,-il.z),rl.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),d._frustum.setFromProjectionMatrix(rl,e.coordinateSystem,e.reversedDepth)}else d.updateMatrices(l);r=d.getFrustum(),b(n,s,d.camera,l,this.type)}d.isPointLightShadow!==!0&&this.type===3&&v(d,s),d.needsUpdate=!1}_=this.type,g.needsUpdate=!1,e.setRenderTarget(c,l,d)};function v(n,r){let a=t.update(h);f.defines.VSM_SAMPLES!==n.blurSamples&&(f.defines.VSM_SAMPLES=n.blurSamples,p.defines.VSM_SAMPLES=n.blurSamples,f.needsUpdate=!0,p.needsUpdate=!0),n.mapPass===null&&(n.mapPass=new Qt(i.x,i.y,{format:me,type:ie})),f.uniforms.shadow_pass.value=n.map.depthTexture,f.uniforms.resolution.value=n.mapSize,f.uniforms.radius.value=n.radius,e.setRenderTarget(n.mapPass),e.clear(),e.renderBufferDirect(r,null,a,f,h,null),p.uniforms.shadow_pass.value=n.mapPass.texture,p.uniforms.resolution.value=n.mapSize,p.uniforms.radius.value=n.radius,e.setRenderTarget(n.map),e.clear(),e.renderBufferDirect(r,null,a,p,h,null)}function y(t,n,r,i){let a=null,o=r.isPointLight===!0?t.customDistanceMaterial:t.customDepthMaterial;if(o!==void 0)a=o;else if(a=r.isPointLight===!0?c:s,e.localClippingEnabled&&n.clipShadows===!0&&Array.isArray(n.clippingPlanes)&&n.clippingPlanes.length!==0||n.displacementMap&&n.displacementScale!==0||n.alphaMap&&n.alphaTest>0||n.map&&n.alphaTest>0||n.alphaToCoverage===!0){let e=a.uuid,t=n.uuid,r=l[e];r===void 0&&(r={},l[e]=r);let i=r[t];i===void 0&&(i=a.clone(),r[t]=i,n.addEventListener(`dispose`,x)),a=i}if(a.visible=n.visible,a.wireframe=n.wireframe,i===3?a.side=n.shadowSide===null?n.side:n.shadowSide:a.side=n.shadowSide===null?d[n.side]:n.shadowSide,a.alphaMap=n.alphaMap,a.alphaTest=n.alphaToCoverage===!0?.5:n.alphaTest,a.map=n.map,a.clipShadows=n.clipShadows,a.clippingPlanes=n.clippingPlanes,a.clipIntersection=n.clipIntersection,a.displacementMap=n.displacementMap,a.displacementScale=n.displacementScale,a.displacementBias=n.displacementBias,a.wireframeLinewidth=n.wireframeLinewidth,a.linewidth=n.linewidth,r.isPointLight===!0&&a.isMeshDistanceMaterial===!0){let t=e.properties.get(a);t.light=r}return a}function b(n,i,a,o,s){if(n.visible===!1)return;if(n.layers.test(i.layers)&&(n.isMesh||n.isLine||n.isPoints)&&(n.castShadow||n.receiveShadow&&s===3)&&(!n.frustumCulled||r.intersectsObject(n))){n.modelViewMatrix.multiplyMatrices(a.matrixWorldInverse,n.matrixWorld);let r=t.update(n),c=n.material;if(Array.isArray(c)){let t=r.groups;for(let l=0,u=t.length;l<u;l++){let u=t[l],d=c[u.materialIndex];if(d&&d.visible){let t=y(n,d,o,s);n.onBeforeShadow(e,n,i,a,r,t,u),e.renderBufferDirect(a,null,r,t,n,u),n.onAfterShadow(e,n,i,a,r,t,u)}}}else if(c.visible){let t=y(n,c,o,s);n.onBeforeShadow(e,n,i,a,r,t,null),e.renderBufferDirect(a,null,r,t,n,null),n.onAfterShadow(e,n,i,a,r,t,null)}}let c=n.children;for(let e=0,t=c.length;e<t;e++)b(c[e],i,a,o,s)}function x(e){e.target.removeEventListener(`dispose`,x);for(let t in l){let n=l[t],r=e.target.uuid;r in n&&(n[r].dispose(),delete n[r])}}}function sl(e,t){function n(){let t=!1,n=new Xt,r=null,i=new Xt(0,0,0,0);return{setMask:function(n){r!==n&&!t&&(e.colorMask(n,n,n,n),r=n)},setLocked:function(e){t=e},setClear:function(t,r,a,o,s){s===!0&&(t*=o,r*=o,a*=o),n.set(t,r,a,o),i.equals(n)===!1&&(e.clearColor(t,r,a,o),i.copy(n))},reset:function(){t=!1,r=null,i.set(-1,0,0,0)}}}function r(){let n=!1,r=!1,i=null,a=null,o=null;return{setReversed:function(e){if(r!==e){let n=t.get(`EXT_clip_control`);e?n.clipControlEXT(n.LOWER_LEFT_EXT,n.ZERO_TO_ONE_EXT):n.clipControlEXT(n.LOWER_LEFT_EXT,n.NEGATIVE_ONE_TO_ONE_EXT),r=e;let i=o;o=null,this.setClear(i)}},getReversed:function(){return r},setTest:function(t){t?z(e.DEPTH_TEST):ae(e.DEPTH_TEST)},setMask:function(t){i!==t&&!n&&(e.depthMask(t),i=t)},setFunc:function(t){if(r&&(t=St[t]),a!==t){switch(t){case 0:e.depthFunc(e.NEVER);break;case 1:e.depthFunc(e.ALWAYS);break;case 2:e.depthFunc(e.LESS);break;case 3:e.depthFunc(e.LEQUAL);break;case 4:e.depthFunc(e.EQUAL);break;case 5:e.depthFunc(e.GEQUAL);break;case 6:e.depthFunc(e.GREATER);break;case 7:e.depthFunc(e.NOTEQUAL);break;default:e.depthFunc(e.LEQUAL)}a=t}},setLocked:function(e){n=e},setClear:function(t){o!==t&&(o=t,r&&(t=1-t),e.clearDepth(t))},reset:function(){n=!1,i=null,a=null,o=null,r=!1}}}function i(){let t=!1,n=null,r=null,i=null,a=null,o=null,s=null,c=null,l=null;return{setTest:function(n){t||(n?z(e.STENCIL_TEST):ae(e.STENCIL_TEST))},setMask:function(r){n!==r&&!t&&(e.stencilMask(r),n=r)},setFunc:function(t,n,o){(r!==t||i!==n||a!==o)&&(e.stencilFunc(t,n,o),r=t,i=n,a=o)},setOp:function(t,n,r){(o!==t||s!==n||c!==r)&&(e.stencilOp(t,n,r),o=t,s=n,c=r)},setLocked:function(e){t=e},setClear:function(t){l!==t&&(e.clearStencil(t),l=t)},reset:function(){t=!1,n=null,r=null,i=null,a=null,o=null,s=null,c=null,l=null}}}let a=new n,o=new r,s=new i,c=new WeakMap,l=new WeakMap,u={},d={},f={},p=new WeakMap,m=[],h=null,g=!1,_=null,v=null,y=null,b=null,x=null,S=null,C=null,w=new Ln(0,0,0),T=0,E=!1,D=null,O=null,k=null,A=null,j=null,M=e.getParameter(e.MAX_COMBINED_TEXTURE_IMAGE_UNITS),N=!1,P=0,F=e.getParameter(e.VERSION);F.indexOf(`WebGL`)===-1?F.indexOf(`OpenGL ES`)!==-1&&(P=parseFloat(/^OpenGL ES (\d)/.exec(F)[1]),N=P>=2):(P=parseFloat(/^WebGL (\d)/.exec(F)[1]),N=P>=1);let I=null,ee={},te=e.getParameter(e.SCISSOR_BOX),L=e.getParameter(e.VIEWPORT),R=new Xt().fromArray(te),ne=new Xt().fromArray(L);function re(t,n,r,i){let a=new Uint8Array(4),o=e.createTexture();e.bindTexture(t,o),e.texParameteri(t,e.TEXTURE_MIN_FILTER,e.NEAREST),e.texParameteri(t,e.TEXTURE_MAG_FILTER,e.NEAREST);for(let o=0;o<r;o++)t===e.TEXTURE_3D||t===e.TEXTURE_2D_ARRAY?e.texImage3D(n,0,e.RGBA,1,1,i,0,e.RGBA,e.UNSIGNED_BYTE,a):e.texImage2D(n+o,0,e.RGBA,1,1,0,e.RGBA,e.UNSIGNED_BYTE,a);return o}let ie={};ie[e.TEXTURE_2D]=re(e.TEXTURE_2D,e.TEXTURE_2D,1),ie[e.TEXTURE_CUBE_MAP]=re(e.TEXTURE_CUBE_MAP,e.TEXTURE_CUBE_MAP_POSITIVE_X,6),ie[e.TEXTURE_2D_ARRAY]=re(e.TEXTURE_2D_ARRAY,e.TEXTURE_2D_ARRAY,1,1),ie[e.TEXTURE_3D]=re(e.TEXTURE_3D,e.TEXTURE_3D,1,1),a.setClear(0,0,0,1),o.setClear(1),s.setClear(0),z(e.DEPTH_TEST),o.setFunc(3),de(!1),fe(1),z(e.CULL_FACE),ue(0);function z(t){u[t]!==!0&&(e.enable(t),u[t]=!0)}function ae(t){u[t]!==!1&&(e.disable(t),u[t]=!1)}function oe(t,n){return f[t]===n?!1:(e.bindFramebuffer(t,n),f[t]=n,t===e.DRAW_FRAMEBUFFER&&(f[e.FRAMEBUFFER]=n),t===e.FRAMEBUFFER&&(f[e.DRAW_FRAMEBUFFER]=n),!0)}function se(t,n){let r=m,i=!1;if(t){r=p.get(n),r===void 0&&(r=[],p.set(n,r));let a=t.textures;if(r.length!==a.length||r[0]!==e.COLOR_ATTACHMENT0){for(let t=0,n=a.length;t<n;t++)r[t]=e.COLOR_ATTACHMENT0+t;r.length=a.length,i=!0}}else r[0]!==e.BACK&&(r[0]=e.BACK,i=!0);i&&e.drawBuffers(r)}function ce(t){return h===t?!1:(e.useProgram(t),h=t,!0)}let le={100:e.FUNC_ADD,101:e.FUNC_SUBTRACT,102:e.FUNC_REVERSE_SUBTRACT};le[103]=e.MIN,le[104]=e.MAX;let B={200:e.ZERO,201:e.ONE,202:e.SRC_COLOR,204:e.SRC_ALPHA,210:e.SRC_ALPHA_SATURATE,208:e.DST_COLOR,206:e.DST_ALPHA,203:e.ONE_MINUS_SRC_COLOR,205:e.ONE_MINUS_SRC_ALPHA,209:e.ONE_MINUS_DST_COLOR,207:e.ONE_MINUS_DST_ALPHA,211:e.CONSTANT_COLOR,212:e.ONE_MINUS_CONSTANT_COLOR,213:e.CONSTANT_ALPHA,214:e.ONE_MINUS_CONSTANT_ALPHA};function ue(t,n,r,i,a,o,s,c,l,u){if(t===0){g===!0&&(ae(e.BLEND),g=!1);return}if(g===!1&&(z(e.BLEND),g=!0),t!==5){if(t!==_||u!==E){if((v!==100||x!==100)&&(e.blendEquation(e.FUNC_ADD),v=100,x=100),u)switch(t){case 1:e.blendFuncSeparate(e.ONE,e.ONE_MINUS_SRC_ALPHA,e.ONE,e.ONE_MINUS_SRC_ALPHA);break;case 2:e.blendFunc(e.ONE,e.ONE);break;case 3:e.blendFuncSeparate(e.ZERO,e.ONE_MINUS_SRC_COLOR,e.ZERO,e.ONE);break;case 4:e.blendFuncSeparate(e.DST_COLOR,e.ONE_MINUS_SRC_ALPHA,e.ZERO,e.ONE);break;default:G(`WebGLState: Invalid blending: `,t);break}else switch(t){case 1:e.blendFuncSeparate(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA,e.ONE,e.ONE_MINUS_SRC_ALPHA);break;case 2:e.blendFuncSeparate(e.SRC_ALPHA,e.ONE,e.ONE,e.ONE);break;case 3:G(`WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true`);break;case 4:G(`WebGLState: MultiplyBlending requires material.premultipliedAlpha = true`);break;default:G(`WebGLState: Invalid blending: `,t);break}y=null,b=null,S=null,C=null,w.set(0,0,0),T=0,_=t,E=u}return}a||=n,o||=r,s||=i,(n!==v||a!==x)&&(e.blendEquationSeparate(le[n],le[a]),v=n,x=a),(r!==y||i!==b||o!==S||s!==C)&&(e.blendFuncSeparate(B[r],B[i],B[o],B[s]),y=r,b=i,S=o,C=s),(c.equals(w)===!1||l!==T)&&(e.blendColor(c.r,c.g,c.b,l),w.copy(c),T=l),_=t,E=!1}function V(t,n){t.side===2?ae(e.CULL_FACE):z(e.CULL_FACE);let r=t.side===1;n&&(r=!r),de(r),t.blending===1&&t.transparent===!1?ue(0):ue(t.blending,t.blendEquation,t.blendSrc,t.blendDst,t.blendEquationAlpha,t.blendSrcAlpha,t.blendDstAlpha,t.blendColor,t.blendAlpha,t.premultipliedAlpha),o.setFunc(t.depthFunc),o.setTest(t.depthTest),o.setMask(t.depthWrite),a.setMask(t.colorWrite);let i=t.stencilWrite;s.setTest(i),i&&(s.setMask(t.stencilWriteMask),s.setFunc(t.stencilFunc,t.stencilRef,t.stencilFuncMask),s.setOp(t.stencilFail,t.stencilZFail,t.stencilZPass)),me(t.polygonOffset,t.polygonOffsetFactor,t.polygonOffsetUnits),t.alphaToCoverage===!0?z(e.SAMPLE_ALPHA_TO_COVERAGE):ae(e.SAMPLE_ALPHA_TO_COVERAGE)}function de(t){D!==t&&(t?e.frontFace(e.CW):e.frontFace(e.CCW),D=t)}function fe(t){t===0?ae(e.CULL_FACE):(z(e.CULL_FACE),t!==O&&(t===1?e.cullFace(e.BACK):t===2?e.cullFace(e.FRONT):e.cullFace(e.FRONT_AND_BACK))),O=t}function pe(t){t!==k&&(N&&e.lineWidth(t),k=t)}function me(t,n,r){t?(z(e.POLYGON_OFFSET_FILL),(A!==n||j!==r)&&(A=n,j=r,o.getReversed()&&(n=-n),e.polygonOffset(n,r))):ae(e.POLYGON_OFFSET_FILL)}function he(t){t?z(e.SCISSOR_TEST):ae(e.SCISSOR_TEST)}function ge(t){t===void 0&&(t=e.TEXTURE0+M-1),I!==t&&(e.activeTexture(t),I=t)}function _e(t,n,r){r===void 0&&(r=I===null?e.TEXTURE0+M-1:I);let i=ee[r];i===void 0&&(i={type:void 0,texture:void 0},ee[r]=i),(i.type!==t||i.texture!==n)&&(I!==r&&(e.activeTexture(r),I=r),e.bindTexture(t,n||ie[t]),i.type=t,i.texture=n)}function ve(){let t=ee[I];t!==void 0&&t.type!==void 0&&(e.bindTexture(t.type,null),t.type=void 0,t.texture=void 0)}function ye(){try{e.compressedTexImage2D(...arguments)}catch(e){G(`WebGLState:`,e)}}function be(){try{e.compressedTexImage3D(...arguments)}catch(e){G(`WebGLState:`,e)}}function xe(){try{e.texSubImage2D(...arguments)}catch(e){G(`WebGLState:`,e)}}function Se(){try{e.texSubImage3D(...arguments)}catch(e){G(`WebGLState:`,e)}}function Ce(){try{e.compressedTexSubImage2D(...arguments)}catch(e){G(`WebGLState:`,e)}}function we(){try{e.compressedTexSubImage3D(...arguments)}catch(e){G(`WebGLState:`,e)}}function Te(){try{e.texStorage2D(...arguments)}catch(e){G(`WebGLState:`,e)}}function Ee(){try{e.texStorage3D(...arguments)}catch(e){G(`WebGLState:`,e)}}function H(){try{e.texImage2D(...arguments)}catch(e){G(`WebGLState:`,e)}}function De(){try{e.texImage3D(...arguments)}catch(e){G(`WebGLState:`,e)}}function Oe(t){return d[t]===void 0?e.getParameter(t):d[t]}function ke(t,n){d[t]!==n&&(e.pixelStorei(t,n),d[t]=n)}function U(t){R.equals(t)===!1&&(e.scissor(t.x,t.y,t.z,t.w),R.copy(t))}function Ae(t){ne.equals(t)===!1&&(e.viewport(t.x,t.y,t.z,t.w),ne.copy(t))}function je(t,n){let r=l.get(n);r===void 0&&(r=new WeakMap,l.set(n,r));let i=r.get(t);i===void 0&&(i=e.getUniformBlockIndex(n,t.name),r.set(t,i))}function Me(t,n){let r=l.get(n).get(t);c.get(n)!==r&&(e.uniformBlockBinding(n,r,t.__bindingPointIndex),c.set(n,r))}function Ne(){e.disable(e.BLEND),e.disable(e.CULL_FACE),e.disable(e.DEPTH_TEST),e.disable(e.POLYGON_OFFSET_FILL),e.disable(e.SCISSOR_TEST),e.disable(e.STENCIL_TEST),e.disable(e.SAMPLE_ALPHA_TO_COVERAGE),e.blendEquation(e.FUNC_ADD),e.blendFunc(e.ONE,e.ZERO),e.blendFuncSeparate(e.ONE,e.ZERO,e.ONE,e.ZERO),e.blendColor(0,0,0,0),e.colorMask(!0,!0,!0,!0),e.clearColor(0,0,0,0),e.depthMask(!0),e.depthFunc(e.LESS),o.setReversed(!1),e.clearDepth(1),e.stencilMask(4294967295),e.stencilFunc(e.ALWAYS,0,4294967295),e.stencilOp(e.KEEP,e.KEEP,e.KEEP),e.clearStencil(0),e.cullFace(e.BACK),e.frontFace(e.CCW),e.polygonOffset(0,0),e.activeTexture(e.TEXTURE0),e.bindFramebuffer(e.FRAMEBUFFER,null),e.bindFramebuffer(e.DRAW_FRAMEBUFFER,null),e.bindFramebuffer(e.READ_FRAMEBUFFER,null),e.useProgram(null),e.lineWidth(1),e.scissor(0,0,e.canvas.width,e.canvas.height),e.viewport(0,0,e.canvas.width,e.canvas.height),e.pixelStorei(e.PACK_ALIGNMENT,4),e.pixelStorei(e.UNPACK_ALIGNMENT,4),e.pixelStorei(e.UNPACK_FLIP_Y_WEBGL,!1),e.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),e.pixelStorei(e.UNPACK_COLORSPACE_CONVERSION_WEBGL,e.BROWSER_DEFAULT_WEBGL),e.pixelStorei(e.PACK_ROW_LENGTH,0),e.pixelStorei(e.PACK_SKIP_PIXELS,0),e.pixelStorei(e.PACK_SKIP_ROWS,0),e.pixelStorei(e.UNPACK_ROW_LENGTH,0),e.pixelStorei(e.UNPACK_IMAGE_HEIGHT,0),e.pixelStorei(e.UNPACK_SKIP_PIXELS,0),e.pixelStorei(e.UNPACK_SKIP_ROWS,0),e.pixelStorei(e.UNPACK_SKIP_IMAGES,0),u={},d={},I=null,ee={},f={},p=new WeakMap,m=[],h=null,g=!1,_=null,v=null,y=null,b=null,x=null,S=null,C=null,w=new Ln(0,0,0),T=0,E=!1,D=null,O=null,k=null,A=null,j=null,R.set(0,0,e.canvas.width,e.canvas.height),ne.set(0,0,e.canvas.width,e.canvas.height),a.reset(),o.reset(),s.reset()}return{buffers:{color:a,depth:o,stencil:s},enable:z,disable:ae,bindFramebuffer:oe,drawBuffers:se,useProgram:ce,setBlending:ue,setMaterial:V,setFlipSided:de,setCullFace:fe,setLineWidth:pe,setPolygonOffset:me,setScissorTest:he,activeTexture:ge,bindTexture:_e,unbindTexture:ve,compressedTexImage2D:ye,compressedTexImage3D:be,texImage2D:H,texImage3D:De,pixelStorei:ke,getParameter:Oe,updateUBOMapping:je,uniformBlockBinding:Me,texStorage2D:Te,texStorage3D:Ee,texSubImage2D:xe,texSubImage3D:Se,compressedTexSubImage2D:Ce,compressedTexSubImage3D:we,scissor:U,viewport:Ae,reset:Ne}}function cl(e,t,n,r,i,a,o){let s=t.has(`WEBGL_multisampled_render_to_texture`)?t.get(`WEBGL_multisampled_render_to_texture`):null,c=typeof navigator>`u`?!1:/OculusBrowser/g.test(navigator.userAgent),l=new Mt,u=new WeakMap,d=new Set,f,p=new WeakMap,m=!1;try{m=typeof OffscreenCanvas<`u`&&new OffscreenCanvas(1,1).getContext(`2d`)!==null}catch{}function h(e,t){return m?new OffscreenCanvas(e,t):mt(`canvas`)}function g(e,t,n){let r=1,i=Oe(e);if((i.width>n||i.height>n)&&(r=n/Math.max(i.width,i.height)),r<1)if(typeof HTMLImageElement<`u`&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<`u`&&e instanceof HTMLCanvasElement||typeof ImageBitmap<`u`&&e instanceof ImageBitmap||typeof VideoFrame<`u`&&e instanceof VideoFrame){let n=Math.floor(r*i.width),a=Math.floor(r*i.height);f===void 0&&(f=h(n,a));let o=t?h(n,a):f;return o.width=n,o.height=a,o.getContext(`2d`).drawImage(e,0,0,n,a),W(`WebGLRenderer: Texture has been resized from (`+i.width+`x`+i.height+`) to (`+n+`x`+a+`).`),o}else return`data`in e&&W(`WebGLRenderer: Image in DataTexture is too big (`+i.width+`x`+i.height+`).`),e;return e}function _(e){return e.generateMipmaps}function v(t){e.generateMipmap(t)}function y(t){return t.isWebGLCubeRenderTarget?e.TEXTURE_CUBE_MAP:t.isWebGL3DRenderTarget?e.TEXTURE_3D:t.isWebGLArrayRenderTarget||t.isCompressedArrayTexture?e.TEXTURE_2D_ARRAY:e.TEXTURE_2D}function b(n,r,i,a,o,s=!1){if(n!==null){if(e[n]!==void 0)return e[n];W(`WebGLRenderer: Attempt to use non-existing WebGL internal format '`+n+`'`)}let c;a&&(c=t.get(`EXT_texture_norm16`),c||W(`WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension`));let l=r;if(r===e.RED&&(i===e.FLOAT&&(l=e.R32F),i===e.HALF_FLOAT&&(l=e.R16F),i===e.UNSIGNED_BYTE&&(l=e.R8),i===e.UNSIGNED_SHORT&&c&&(l=c.R16_EXT),i===e.SHORT&&c&&(l=c.R16_SNORM_EXT)),r===e.RED_INTEGER&&(i===e.UNSIGNED_BYTE&&(l=e.R8UI),i===e.UNSIGNED_SHORT&&(l=e.R16UI),i===e.UNSIGNED_INT&&(l=e.R32UI),i===e.BYTE&&(l=e.R8I),i===e.SHORT&&(l=e.R16I),i===e.INT&&(l=e.R32I)),r===e.RG&&(i===e.FLOAT&&(l=e.RG32F),i===e.HALF_FLOAT&&(l=e.RG16F),i===e.UNSIGNED_BYTE&&(l=e.RG8),i===e.UNSIGNED_SHORT&&c&&(l=c.RG16_EXT),i===e.SHORT&&c&&(l=c.RG16_SNORM_EXT)),r===e.RG_INTEGER&&(i===e.UNSIGNED_BYTE&&(l=e.RG8UI),i===e.UNSIGNED_SHORT&&(l=e.RG16UI),i===e.UNSIGNED_INT&&(l=e.RG32UI),i===e.BYTE&&(l=e.RG8I),i===e.SHORT&&(l=e.RG16I),i===e.INT&&(l=e.RG32I)),r===e.RGB_INTEGER&&(i===e.UNSIGNED_BYTE&&(l=e.RGB8UI),i===e.UNSIGNED_SHORT&&(l=e.RGB16UI),i===e.UNSIGNED_INT&&(l=e.RGB32UI),i===e.BYTE&&(l=e.RGB8I),i===e.SHORT&&(l=e.RGB16I),i===e.INT&&(l=e.RGB32I)),r===e.RGBA_INTEGER&&(i===e.UNSIGNED_BYTE&&(l=e.RGBA8UI),i===e.UNSIGNED_SHORT&&(l=e.RGBA16UI),i===e.UNSIGNED_INT&&(l=e.RGBA32UI),i===e.BYTE&&(l=e.RGBA8I),i===e.SHORT&&(l=e.RGBA16I),i===e.INT&&(l=e.RGBA32I)),r===e.RGB&&(i===e.UNSIGNED_SHORT&&c&&(l=c.RGB16_EXT),i===e.SHORT&&c&&(l=c.RGB16_SNORM_EXT),i===e.UNSIGNED_INT_5_9_9_9_REV&&(l=e.RGB9_E5),i===e.UNSIGNED_INT_10F_11F_11F_REV&&(l=e.R11F_G11F_B10F)),r===e.RGBA){let t=s?st:Y.getTransfer(o);i===e.FLOAT&&(l=e.RGBA32F),i===e.HALF_FLOAT&&(l=e.RGBA16F),i===e.UNSIGNED_BYTE&&(l=t===`srgb`?e.SRGB8_ALPHA8:e.RGBA8),i===e.UNSIGNED_SHORT&&c&&(l=c.RGBA16_EXT),i===e.SHORT&&c&&(l=c.RGBA16_SNORM_EXT),i===e.UNSIGNED_SHORT_4_4_4_4&&(l=e.RGBA4),i===e.UNSIGNED_SHORT_5_5_5_1&&(l=e.RGB5_A1)}return(l===e.R16F||l===e.R32F||l===e.RG16F||l===e.RG32F||l===e.RGBA16F||l===e.RGBA32F)&&t.get(`EXT_color_buffer_float`),l}function x(t,n){let r;return t?n===null||n===1014||n===1020?r=e.DEPTH24_STENCIL8:n===1015?r=e.DEPTH32F_STENCIL8:n===1012&&(r=e.DEPTH24_STENCIL8,W(`DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.`)):n===null||n===1014||n===1020?r=e.DEPTH_COMPONENT24:n===1015?r=e.DEPTH_COMPONENT32F:n===1012&&(r=e.DEPTH_COMPONENT16),r}function S(e,t){return _(e)===!0||e.isFramebufferTexture&&e.minFilter!==1003&&e.minFilter!==1006?Math.log2(Math.max(t.width,t.height))+1:e.mipmaps!==void 0&&e.mipmaps.length>0?e.mipmaps.length:e.isCompressedTexture&&Array.isArray(e.image)?t.mipmaps.length:1}function C(e){let t=e.target;t.removeEventListener(`dispose`,C),T(t),t.isVideoTexture&&u.delete(t),t.isHTMLTexture&&d.delete(t)}function w(e){let t=e.target;t.removeEventListener(`dispose`,w),I(t)}function T(e){let t=r.get(e);if(t.__webglInit===void 0)return;let n=e.source,i=p.get(n);if(i){let r=i[t.__cacheKey];r.usedTimes--,r.usedTimes===0&&E(e),Object.keys(i).length===0&&p.delete(n)}r.remove(e)}function E(t){let n=r.get(t);e.deleteTexture(n.__webglTexture);let i=t.source,a=p.get(i);delete a[n.__cacheKey],o.memory.textures--}function I(t){let n=r.get(t);if(t.depthTexture&&(t.depthTexture.dispose(),r.remove(t.depthTexture)),t.isWebGLCubeRenderTarget)for(let t=0;t<6;t++){if(Array.isArray(n.__webglFramebuffer[t]))for(let r=0;r<n.__webglFramebuffer[t].length;r++)e.deleteFramebuffer(n.__webglFramebuffer[t][r]);else e.deleteFramebuffer(n.__webglFramebuffer[t]);n.__webglDepthbuffer&&e.deleteRenderbuffer(n.__webglDepthbuffer[t])}else{if(Array.isArray(n.__webglFramebuffer))for(let t=0;t<n.__webglFramebuffer.length;t++)e.deleteFramebuffer(n.__webglFramebuffer[t]);else e.deleteFramebuffer(n.__webglFramebuffer);if(n.__webglDepthbuffer&&e.deleteRenderbuffer(n.__webglDepthbuffer),n.__webglMultisampledFramebuffer&&e.deleteFramebuffer(n.__webglMultisampledFramebuffer),n.__webglColorRenderbuffer)for(let t=0;t<n.__webglColorRenderbuffer.length;t++)n.__webglColorRenderbuffer[t]&&e.deleteRenderbuffer(n.__webglColorRenderbuffer[t]);n.__webglDepthRenderbuffer&&e.deleteRenderbuffer(n.__webglDepthRenderbuffer)}let i=t.textures;for(let t=0,n=i.length;t<n;t++){let n=r.get(i[t]);n.__webglTexture&&(e.deleteTexture(n.__webglTexture),o.memory.textures--),r.remove(i[t])}r.remove(t)}let ee=0;function te(){ee=0}function L(){return ee}function R(e){ee=e}function ne(){let e=ee;return e>=i.maxTextures&&W(`WebGLTextures: Trying to use `+e+` texture units while this GPU supports only `+i.maxTextures),ee+=1,e}function re(e){let t=[];return t.push(e.wrapS),t.push(e.wrapT),t.push(e.wrapR||0),t.push(e.magFilter),t.push(e.minFilter),t.push(e.anisotropy),t.push(e.internalFormat),t.push(e.format),t.push(e.type),t.push(e.generateMipmaps),t.push(e.premultiplyAlpha),t.push(e.flipY),t.push(e.unpackAlignment),t.push(e.colorSpace),t.join()}function ie(t,i){let a=r.get(t);if(t.isVideoTexture&&H(t),t.isRenderTargetTexture===!1&&t.isExternalTexture!==!0&&t.version>0&&a.__version!==t.version){let e=t.image;if(e===null)W(`WebGLRenderer: Texture marked for update but no image data found.`);else if(e.complete===!1)W(`WebGLRenderer: Texture marked for update but image is incomplete`);else{pe(a,t,i);return}}else t.isExternalTexture&&(a.__webglTexture=t.sourceTexture?t.sourceTexture:null);n.bindTexture(e.TEXTURE_2D,a.__webglTexture,e.TEXTURE0+i)}function z(t,i){let a=r.get(t);if(t.isRenderTargetTexture===!1&&t.version>0&&a.__version!==t.version){pe(a,t,i);return}else t.isExternalTexture&&(a.__webglTexture=t.sourceTexture?t.sourceTexture:null);n.bindTexture(e.TEXTURE_2D_ARRAY,a.__webglTexture,e.TEXTURE0+i)}function ae(t,i){let a=r.get(t);if(t.isRenderTargetTexture===!1&&t.version>0&&a.__version!==t.version){pe(a,t,i);return}n.bindTexture(e.TEXTURE_3D,a.__webglTexture,e.TEXTURE0+i)}function oe(t,i){let a=r.get(t);if(t.isCubeDepthTexture!==!0&&t.version>0&&a.__version!==t.version){me(a,t,i);return}n.bindTexture(e.TEXTURE_CUBE_MAP,a.__webglTexture,e.TEXTURE0+i)}let se={[D]:e.REPEAT,[O]:e.CLAMP_TO_EDGE,[k]:e.MIRRORED_REPEAT},ce={[A]:e.NEAREST,[j]:e.NEAREST_MIPMAP_NEAREST,[M]:e.NEAREST_MIPMAP_LINEAR,[N]:e.LINEAR,[P]:e.LINEAR_MIPMAP_NEAREST,[F]:e.LINEAR_MIPMAP_LINEAR},le={512:e.NEVER,519:e.ALWAYS,513:e.LESS,515:e.LEQUAL,514:e.EQUAL,518:e.GEQUAL,516:e.GREATER,517:e.NOTEQUAL};function B(n,a){if(a.type===1015&&t.has(`OES_texture_float_linear`)===!1&&(a.magFilter===1006||a.magFilter===1007||a.magFilter===1005||a.magFilter===1008||a.minFilter===1006||a.minFilter===1007||a.minFilter===1005||a.minFilter===1008)&&W(`WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device.`),e.texParameteri(n,e.TEXTURE_WRAP_S,se[a.wrapS]),e.texParameteri(n,e.TEXTURE_WRAP_T,se[a.wrapT]),(n===e.TEXTURE_3D||n===e.TEXTURE_2D_ARRAY)&&e.texParameteri(n,e.TEXTURE_WRAP_R,se[a.wrapR]),e.texParameteri(n,e.TEXTURE_MAG_FILTER,ce[a.magFilter]),e.texParameteri(n,e.TEXTURE_MIN_FILTER,ce[a.minFilter]),a.compareFunction&&(e.texParameteri(n,e.TEXTURE_COMPARE_MODE,e.COMPARE_REF_TO_TEXTURE),e.texParameteri(n,e.TEXTURE_COMPARE_FUNC,le[a.compareFunction])),t.has(`EXT_texture_filter_anisotropic`)===!0){if(a.magFilter===1003||a.minFilter!==1005&&a.minFilter!==1008||a.type===1015&&t.has(`OES_texture_float_linear`)===!1)return;if(a.anisotropy>1||r.get(a).__currentAnisotropy){let o=t.get(`EXT_texture_filter_anisotropic`);e.texParameterf(n,o.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(a.anisotropy,i.getMaxAnisotropy())),r.get(a).__currentAnisotropy=a.anisotropy}}}function ue(t,n){let r=!1;t.__webglInit===void 0&&(t.__webglInit=!0,n.addEventListener(`dispose`,C));let i=n.source,a=p.get(i);a===void 0&&(a={},p.set(i,a));let s=re(n);if(s!==t.__cacheKey){a[s]===void 0&&(a[s]={texture:e.createTexture(),usedTimes:0},o.memory.textures++,r=!0),a[s].usedTimes++;let i=a[t.__cacheKey];i!==void 0&&(a[t.__cacheKey].usedTimes--,i.usedTimes===0&&E(n)),t.__cacheKey=s,t.__webglTexture=a[s].texture}return r}function V(e,t,n){return Math.floor(Math.floor(e/n)/t)}function fe(t,r,i,a){let o=t.updateRanges;if(o.length===0)n.texSubImage2D(e.TEXTURE_2D,0,0,0,r.width,r.height,i,a,r.data);else{o.sort((e,t)=>e.start-t.start);let s=0;for(let e=1;e<o.length;e++){let t=o[s],n=o[e],i=t.start+t.count,a=V(n.start,r.width,4),c=V(t.start,r.width,4);n.start<=i+1&&a===c&&V(n.start+n.count-1,r.width,4)===a?t.count=Math.max(t.count,n.start+n.count-t.start):(++s,o[s]=n)}o.length=s+1;let c=n.getParameter(e.UNPACK_ROW_LENGTH),l=n.getParameter(e.UNPACK_SKIP_PIXELS),u=n.getParameter(e.UNPACK_SKIP_ROWS);n.pixelStorei(e.UNPACK_ROW_LENGTH,r.width);for(let t=0,s=o.length;t<s;t++){let s=o[t],c=Math.floor(s.start/4),l=Math.ceil(s.count/4),u=c%r.width,d=Math.floor(c/r.width),f=l;n.pixelStorei(e.UNPACK_SKIP_PIXELS,u),n.pixelStorei(e.UNPACK_SKIP_ROWS,d),n.texSubImage2D(e.TEXTURE_2D,0,u,d,f,1,i,a,r.data)}t.clearUpdateRanges(),n.pixelStorei(e.UNPACK_ROW_LENGTH,c),n.pixelStorei(e.UNPACK_SKIP_PIXELS,l),n.pixelStorei(e.UNPACK_SKIP_ROWS,u)}}function pe(t,o,s){let c=e.TEXTURE_2D;(o.isDataArrayTexture||o.isCompressedArrayTexture)&&(c=e.TEXTURE_2D_ARRAY),o.isData3DTexture&&(c=e.TEXTURE_3D);let l=ue(t,o),u=o.source;n.bindTexture(c,t.__webglTexture,e.TEXTURE0+s);let f=r.get(u);if(u.version!==f.__version||l===!0){if(n.activeTexture(e.TEXTURE0+s),!(typeof ImageBitmap<`u`&&o.image instanceof ImageBitmap)){let t=Y.getPrimaries(Y.workingColorSpace),r=o.colorSpace===``?null:Y.getPrimaries(o.colorSpace),i=o.colorSpace===``||t===r?e.NONE:e.BROWSER_DEFAULT_WEBGL;n.pixelStorei(e.UNPACK_FLIP_Y_WEBGL,o.flipY),n.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL,o.premultiplyAlpha),n.pixelStorei(e.UNPACK_COLORSPACE_CONVERSION_WEBGL,i)}n.pixelStorei(e.UNPACK_ALIGNMENT,o.unpackAlignment);let t=g(o.image,!1,i.maxTextureSize);t=De(o,t);let r=a.convert(o.format,o.colorSpace),p=a.convert(o.type),m=b(o.internalFormat,r,p,o.normalized,o.colorSpace,o.isVideoTexture);B(c,o);let h,y=o.mipmaps,C=o.isVideoTexture!==!0,w=f.__version===void 0||l===!0,T=u.dataReady,E=S(o,t);if(o.isDepthTexture)m=x(o.format===de,o.type),w&&(C?n.texStorage2D(e.TEXTURE_2D,1,m,t.width,t.height):n.texImage2D(e.TEXTURE_2D,0,m,t.width,t.height,0,r,p,null));else if(o.isDataTexture)if(y.length>0){C&&w&&n.texStorage2D(e.TEXTURE_2D,E,m,y[0].width,y[0].height);for(let t=0,i=y.length;t<i;t++)h=y[t],C?T&&n.texSubImage2D(e.TEXTURE_2D,t,0,0,h.width,h.height,r,p,h.data):n.texImage2D(e.TEXTURE_2D,t,m,h.width,h.height,0,r,p,h.data);o.generateMipmaps=!1}else C?(w&&n.texStorage2D(e.TEXTURE_2D,E,m,t.width,t.height),T&&fe(o,t,r,p)):n.texImage2D(e.TEXTURE_2D,0,m,t.width,t.height,0,r,p,t.data);else if(o.isCompressedTexture)if(o.isCompressedArrayTexture){C&&w&&n.texStorage3D(e.TEXTURE_2D_ARRAY,E,m,y[0].width,y[0].height,t.depth);for(let i=0,a=y.length;i<a;i++)if(h=y[i],o.format!==1023)if(r!==null)if(C){if(T)if(o.layerUpdates.size>0){let t=no(h.width,h.height,o.format,o.type);for(let a of o.layerUpdates){let o=h.data.subarray(a*t/h.data.BYTES_PER_ELEMENT,(a+1)*t/h.data.BYTES_PER_ELEMENT);n.compressedTexSubImage3D(e.TEXTURE_2D_ARRAY,i,0,0,a,h.width,h.height,1,r,o)}o.clearLayerUpdates()}else n.compressedTexSubImage3D(e.TEXTURE_2D_ARRAY,i,0,0,0,h.width,h.height,t.depth,r,h.data)}else n.compressedTexImage3D(e.TEXTURE_2D_ARRAY,i,m,h.width,h.height,t.depth,0,h.data,0,0);else W(`WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()`);else C?T&&n.texSubImage3D(e.TEXTURE_2D_ARRAY,i,0,0,0,h.width,h.height,t.depth,r,p,h.data):n.texImage3D(e.TEXTURE_2D_ARRAY,i,m,h.width,h.height,t.depth,0,r,p,h.data)}else{C&&w&&n.texStorage2D(e.TEXTURE_2D,E,m,y[0].width,y[0].height);for(let t=0,i=y.length;t<i;t++)h=y[t],o.format===1023?C?T&&n.texSubImage2D(e.TEXTURE_2D,t,0,0,h.width,h.height,r,p,h.data):n.texImage2D(e.TEXTURE_2D,t,m,h.width,h.height,0,r,p,h.data):r===null?W(`WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()`):C?T&&n.compressedTexSubImage2D(e.TEXTURE_2D,t,0,0,h.width,h.height,r,h.data):n.compressedTexImage2D(e.TEXTURE_2D,t,m,h.width,h.height,0,h.data)}else if(o.isDataArrayTexture)if(C){if(w&&n.texStorage3D(e.TEXTURE_2D_ARRAY,E,m,t.width,t.height,t.depth),T)if(o.layerUpdates.size>0){let i=no(t.width,t.height,o.format,o.type);for(let a of o.layerUpdates){let o=t.data.subarray(a*i/t.data.BYTES_PER_ELEMENT,(a+1)*i/t.data.BYTES_PER_ELEMENT);n.texSubImage3D(e.TEXTURE_2D_ARRAY,0,0,0,a,t.width,t.height,1,r,p,o)}o.clearLayerUpdates()}else n.texSubImage3D(e.TEXTURE_2D_ARRAY,0,0,0,0,t.width,t.height,t.depth,r,p,t.data)}else n.texImage3D(e.TEXTURE_2D_ARRAY,0,m,t.width,t.height,t.depth,0,r,p,t.data);else if(o.isData3DTexture)C?(w&&n.texStorage3D(e.TEXTURE_3D,E,m,t.width,t.height,t.depth),T&&n.texSubImage3D(e.TEXTURE_3D,0,0,0,0,t.width,t.height,t.depth,r,p,t.data)):n.texImage3D(e.TEXTURE_3D,0,m,t.width,t.height,t.depth,0,r,p,t.data);else if(o.isFramebufferTexture){if(w)if(C)n.texStorage2D(e.TEXTURE_2D,E,m,t.width,t.height);else{let i=t.width,a=t.height;for(let t=0;t<E;t++)n.texImage2D(e.TEXTURE_2D,t,m,i,a,0,r,p,null),i>>=1,a>>=1}}else if(o.isHTMLTexture){if(`texElementImage2D`in e){let n=e.canvas;if(n.hasAttribute(`layoutsubtree`)||n.setAttribute(`layoutsubtree`,`true`),t.parentNode!==n){n.appendChild(t),d.add(o),n.onpaint=e=>{let t=e.changedElements;for(let e of d)t.includes(e.image)&&(e.needsUpdate=!0)},n.requestPaint();return}if(e.texElementImage2D.length===3)e.texElementImage2D(e.TEXTURE_2D,e.RGBA8,t);else{let n=e.RGBA,r=e.RGBA,i=e.UNSIGNED_BYTE;e.texElementImage2D(e.TEXTURE_2D,0,n,r,i,t)}e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MIN_FILTER,e.LINEAR),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE)}}else if(y.length>0){if(C&&w){let t=Oe(y[0]);n.texStorage2D(e.TEXTURE_2D,E,m,t.width,t.height)}for(let t=0,i=y.length;t<i;t++)h=y[t],C?T&&n.texSubImage2D(e.TEXTURE_2D,t,0,0,r,p,h):n.texImage2D(e.TEXTURE_2D,t,m,r,p,h);o.generateMipmaps=!1}else if(C){if(w){let r=Oe(t);n.texStorage2D(e.TEXTURE_2D,E,m,r.width,r.height)}T&&n.texSubImage2D(e.TEXTURE_2D,0,0,0,r,p,t)}else n.texImage2D(e.TEXTURE_2D,0,m,r,p,t);_(o)&&v(c),f.__version=u.version,o.onUpdate&&o.onUpdate(o)}t.__version=o.version}function me(t,o,s){if(o.image.length!==6)return;let c=ue(t,o),l=o.source;n.bindTexture(e.TEXTURE_CUBE_MAP,t.__webglTexture,e.TEXTURE0+s);let u=r.get(l);if(l.version!==u.__version||c===!0){n.activeTexture(e.TEXTURE0+s);let t=Y.getPrimaries(Y.workingColorSpace),r=o.colorSpace===``?null:Y.getPrimaries(o.colorSpace),d=o.colorSpace===``||t===r?e.NONE:e.BROWSER_DEFAULT_WEBGL;n.pixelStorei(e.UNPACK_FLIP_Y_WEBGL,o.flipY),n.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL,o.premultiplyAlpha),n.pixelStorei(e.UNPACK_ALIGNMENT,o.unpackAlignment),n.pixelStorei(e.UNPACK_COLORSPACE_CONVERSION_WEBGL,d);let f=o.isCompressedTexture||o.image[0].isCompressedTexture,p=o.image[0]&&o.image[0].isDataTexture,m=[];for(let e=0;e<6;e++)!f&&!p?m[e]=g(o.image[e],!0,i.maxCubemapSize):m[e]=p?o.image[e].image:o.image[e],m[e]=De(o,m[e]);let h=m[0],y=a.convert(o.format,o.colorSpace),x=a.convert(o.type),C=b(o.internalFormat,y,x,o.normalized,o.colorSpace),w=o.isVideoTexture!==!0,T=u.__version===void 0||c===!0,E=l.dataReady,D=S(o,h);B(e.TEXTURE_CUBE_MAP,o);let O;if(f){w&&T&&n.texStorage2D(e.TEXTURE_CUBE_MAP,D,C,h.width,h.height);for(let t=0;t<6;t++){O=m[t].mipmaps;for(let r=0;r<O.length;r++){let i=O[r];o.format===1023?w?E&&n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r,0,0,i.width,i.height,y,x,i.data):n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r,C,i.width,i.height,0,y,x,i.data):y===null?W(`WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()`):w?E&&n.compressedTexSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r,0,0,i.width,i.height,y,i.data):n.compressedTexImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r,C,i.width,i.height,0,i.data)}}}else{if(O=o.mipmaps,w&&T){O.length>0&&D++;let t=Oe(m[0]);n.texStorage2D(e.TEXTURE_CUBE_MAP,D,C,t.width,t.height)}for(let t=0;t<6;t++)if(p){w?E&&n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,0,0,0,m[t].width,m[t].height,y,x,m[t].data):n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,0,C,m[t].width,m[t].height,0,y,x,m[t].data);for(let r=0;r<O.length;r++){let i=O[r].image[t].image;w?E&&n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r+1,0,0,i.width,i.height,y,x,i.data):n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r+1,C,i.width,i.height,0,y,x,i.data)}}else{w?E&&n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,0,0,0,y,x,m[t]):n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,0,C,y,x,m[t]);for(let r=0;r<O.length;r++){let i=O[r];w?E&&n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r+1,0,0,y,x,i.image[t]):n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+t,r+1,C,y,x,i.image[t])}}}_(o)&&v(e.TEXTURE_CUBE_MAP),u.__version=l.version,o.onUpdate&&o.onUpdate(o)}t.__version=o.version}function he(t,i,o,c,l,u){let d=a.convert(o.format,o.colorSpace),f=a.convert(o.type),p=b(o.internalFormat,d,f,o.normalized,o.colorSpace),m=r.get(i),h=r.get(o);if(h.__renderTarget=i,!m.__hasExternalTextures){let t=Math.max(1,i.width>>u),r=Math.max(1,i.height>>u);l===e.TEXTURE_3D||l===e.TEXTURE_2D_ARRAY?n.texImage3D(l,u,p,t,r,i.depth,0,d,f,null):n.texImage2D(l,u,p,t,r,0,d,f,null)}n.bindFramebuffer(e.FRAMEBUFFER,t),Ee(i)?s.framebufferTexture2DMultisampleEXT(e.FRAMEBUFFER,c,l,h.__webglTexture,0,Te(i)):(l===e.TEXTURE_2D||l>=e.TEXTURE_CUBE_MAP_POSITIVE_X&&l<=e.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&e.framebufferTexture2D(e.FRAMEBUFFER,c,l,h.__webglTexture,u),n.bindFramebuffer(e.FRAMEBUFFER,null)}function ge(t,n,r){if(e.bindRenderbuffer(e.RENDERBUFFER,t),n.depthBuffer){let i=n.depthTexture,a=i&&i.isDepthTexture?i.type:null,o=x(n.stencilBuffer,a),c=n.stencilBuffer?e.DEPTH_STENCIL_ATTACHMENT:e.DEPTH_ATTACHMENT;Ee(n)?s.renderbufferStorageMultisampleEXT(e.RENDERBUFFER,Te(n),o,n.width,n.height):r?e.renderbufferStorageMultisample(e.RENDERBUFFER,Te(n),o,n.width,n.height):e.renderbufferStorage(e.RENDERBUFFER,o,n.width,n.height),e.framebufferRenderbuffer(e.FRAMEBUFFER,c,e.RENDERBUFFER,t)}else{let t=n.textures;for(let i=0;i<t.length;i++){let o=t[i],c=a.convert(o.format,o.colorSpace),l=a.convert(o.type),u=b(o.internalFormat,c,l,o.normalized,o.colorSpace);Ee(n)?s.renderbufferStorageMultisampleEXT(e.RENDERBUFFER,Te(n),u,n.width,n.height):r?e.renderbufferStorageMultisample(e.RENDERBUFFER,Te(n),u,n.width,n.height):e.renderbufferStorage(e.RENDERBUFFER,u,n.width,n.height)}}e.bindRenderbuffer(e.RENDERBUFFER,null)}function _e(t,i,o){let c=i.isWebGLCubeRenderTarget===!0;if(n.bindFramebuffer(e.FRAMEBUFFER,t),!(i.depthTexture&&i.depthTexture.isDepthTexture))throw Error(`THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.`);let l=r.get(i.depthTexture);if(l.__renderTarget=i,(!l.__webglTexture||i.depthTexture.image.width!==i.width||i.depthTexture.image.height!==i.height)&&(i.depthTexture.image.width=i.width,i.depthTexture.image.height=i.height,i.depthTexture.needsUpdate=!0),c){if(l.__webglInit===void 0&&(l.__webglInit=!0,i.depthTexture.addEventListener(`dispose`,C)),l.__webglTexture===void 0){l.__webglTexture=e.createTexture(),n.bindTexture(e.TEXTURE_CUBE_MAP,l.__webglTexture),B(e.TEXTURE_CUBE_MAP,i.depthTexture);let t=a.convert(i.depthTexture.format),r=a.convert(i.depthTexture.type),o;i.depthTexture.format===1026?o=e.DEPTH_COMPONENT24:i.depthTexture.format===1027&&(o=e.DEPTH24_STENCIL8);for(let n=0;n<6;n++)e.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X+n,0,o,i.width,i.height,0,t,r,null)}}else ie(i.depthTexture,0);let u=l.__webglTexture,d=Te(i),f=c?e.TEXTURE_CUBE_MAP_POSITIVE_X+o:e.TEXTURE_2D,p=i.depthTexture.format===1027?e.DEPTH_STENCIL_ATTACHMENT:e.DEPTH_ATTACHMENT;if(i.depthTexture.format===1026)Ee(i)?s.framebufferTexture2DMultisampleEXT(e.FRAMEBUFFER,p,f,u,0,d):e.framebufferTexture2D(e.FRAMEBUFFER,p,f,u,0);else if(i.depthTexture.format===1027)Ee(i)?s.framebufferTexture2DMultisampleEXT(e.FRAMEBUFFER,p,f,u,0,d):e.framebufferTexture2D(e.FRAMEBUFFER,p,f,u,0);else throw Error(`THREE.WebGLTextures: Unknown depthTexture format.`)}function ve(t){let i=r.get(t),a=t.isWebGLCubeRenderTarget===!0;if(i.__boundDepthTexture!==t.depthTexture){let e=t.depthTexture;if(i.__depthDisposeCallback&&i.__depthDisposeCallback(),e){let t=()=>{delete i.__boundDepthTexture,delete i.__depthDisposeCallback,e.removeEventListener(`dispose`,t)};e.addEventListener(`dispose`,t),i.__depthDisposeCallback=t}i.__boundDepthTexture=e}if(t.depthTexture&&!i.__autoAllocateDepthBuffer)if(a)for(let e=0;e<6;e++)_e(i.__webglFramebuffer[e],t,e);else{let e=t.texture.mipmaps;e&&e.length>0?_e(i.__webglFramebuffer[0],t,0):_e(i.__webglFramebuffer,t,0)}else if(a){i.__webglDepthbuffer=[];for(let r=0;r<6;r++)if(n.bindFramebuffer(e.FRAMEBUFFER,i.__webglFramebuffer[r]),i.__webglDepthbuffer[r]===void 0)i.__webglDepthbuffer[r]=e.createRenderbuffer(),ge(i.__webglDepthbuffer[r],t,!1);else{let n=t.stencilBuffer?e.DEPTH_STENCIL_ATTACHMENT:e.DEPTH_ATTACHMENT,a=i.__webglDepthbuffer[r];e.bindRenderbuffer(e.RENDERBUFFER,a),e.framebufferRenderbuffer(e.FRAMEBUFFER,n,e.RENDERBUFFER,a)}}else{let r=t.texture.mipmaps;if(r&&r.length>0?n.bindFramebuffer(e.FRAMEBUFFER,i.__webglFramebuffer[0]):n.bindFramebuffer(e.FRAMEBUFFER,i.__webglFramebuffer),i.__webglDepthbuffer===void 0)i.__webglDepthbuffer=e.createRenderbuffer(),ge(i.__webglDepthbuffer,t,!1);else{let n=t.stencilBuffer?e.DEPTH_STENCIL_ATTACHMENT:e.DEPTH_ATTACHMENT,r=i.__webglDepthbuffer;e.bindRenderbuffer(e.RENDERBUFFER,r),e.framebufferRenderbuffer(e.FRAMEBUFFER,n,e.RENDERBUFFER,r)}}n.bindFramebuffer(e.FRAMEBUFFER,null)}function ye(t,n,i){let a=r.get(t);n!==void 0&&he(a.__webglFramebuffer,t,t.texture,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,0),i!==void 0&&ve(t)}function be(t){let i=t.texture,s=r.get(t),c=r.get(i);t.addEventListener(`dispose`,w);let l=t.textures,u=t.isWebGLCubeRenderTarget===!0,d=l.length>1;if(d||(c.__webglTexture===void 0&&(c.__webglTexture=e.createTexture()),c.__version=i.version,o.memory.textures++),u){s.__webglFramebuffer=[];for(let t=0;t<6;t++)if(i.mipmaps&&i.mipmaps.length>0){s.__webglFramebuffer[t]=[];for(let n=0;n<i.mipmaps.length;n++)s.__webglFramebuffer[t][n]=e.createFramebuffer()}else s.__webglFramebuffer[t]=e.createFramebuffer()}else{if(i.mipmaps&&i.mipmaps.length>0){s.__webglFramebuffer=[];for(let t=0;t<i.mipmaps.length;t++)s.__webglFramebuffer[t]=e.createFramebuffer()}else s.__webglFramebuffer=e.createFramebuffer();if(d)for(let t=0,n=l.length;t<n;t++){let n=r.get(l[t]);n.__webglTexture===void 0&&(n.__webglTexture=e.createTexture(),o.memory.textures++)}if(t.samples>0&&Ee(t)===!1){s.__webglMultisampledFramebuffer=e.createFramebuffer(),s.__webglColorRenderbuffer=[],n.bindFramebuffer(e.FRAMEBUFFER,s.__webglMultisampledFramebuffer);for(let n=0;n<l.length;n++){let r=l[n];s.__webglColorRenderbuffer[n]=e.createRenderbuffer(),e.bindRenderbuffer(e.RENDERBUFFER,s.__webglColorRenderbuffer[n]);let i=a.convert(r.format,r.colorSpace),o=a.convert(r.type),c=b(r.internalFormat,i,o,r.normalized,r.colorSpace,t.isXRRenderTarget===!0),u=Te(t);e.renderbufferStorageMultisample(e.RENDERBUFFER,u,c,t.width,t.height),e.framebufferRenderbuffer(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0+n,e.RENDERBUFFER,s.__webglColorRenderbuffer[n])}e.bindRenderbuffer(e.RENDERBUFFER,null),t.depthBuffer&&(s.__webglDepthRenderbuffer=e.createRenderbuffer(),ge(s.__webglDepthRenderbuffer,t,!0)),n.bindFramebuffer(e.FRAMEBUFFER,null)}}if(u){n.bindTexture(e.TEXTURE_CUBE_MAP,c.__webglTexture),B(e.TEXTURE_CUBE_MAP,i);for(let n=0;n<6;n++)if(i.mipmaps&&i.mipmaps.length>0)for(let r=0;r<i.mipmaps.length;r++)he(s.__webglFramebuffer[n][r],t,i,e.COLOR_ATTACHMENT0,e.TEXTURE_CUBE_MAP_POSITIVE_X+n,r);else he(s.__webglFramebuffer[n],t,i,e.COLOR_ATTACHMENT0,e.TEXTURE_CUBE_MAP_POSITIVE_X+n,0);_(i)&&v(e.TEXTURE_CUBE_MAP),n.unbindTexture()}else if(d){for(let i=0,a=l.length;i<a;i++){let a=l[i],o=r.get(a),c=e.TEXTURE_2D;(t.isWebGL3DRenderTarget||t.isWebGLArrayRenderTarget)&&(c=t.isWebGL3DRenderTarget?e.TEXTURE_3D:e.TEXTURE_2D_ARRAY),n.bindTexture(c,o.__webglTexture),B(c,a),he(s.__webglFramebuffer,t,a,e.COLOR_ATTACHMENT0+i,c,0),_(a)&&v(c)}n.unbindTexture()}else{let r=e.TEXTURE_2D;if((t.isWebGL3DRenderTarget||t.isWebGLArrayRenderTarget)&&(r=t.isWebGL3DRenderTarget?e.TEXTURE_3D:e.TEXTURE_2D_ARRAY),n.bindTexture(r,c.__webglTexture),B(r,i),i.mipmaps&&i.mipmaps.length>0)for(let n=0;n<i.mipmaps.length;n++)he(s.__webglFramebuffer[n],t,i,e.COLOR_ATTACHMENT0,r,n);else he(s.__webglFramebuffer,t,i,e.COLOR_ATTACHMENT0,r,0);_(i)&&v(r),n.unbindTexture()}t.depthBuffer&&ve(t)}function xe(e){let t=e.textures;for(let i=0,a=t.length;i<a;i++){let a=t[i];if(_(a)){let t=y(e),i=r.get(a).__webglTexture;n.bindTexture(t,i),v(t),n.unbindTexture()}}}let Se=[],Ce=[];function we(t){if(t.samples>0){if(Ee(t)===!1){let i=t.textures,a=t.width,o=t.height,s=e.COLOR_BUFFER_BIT,l=t.stencilBuffer?e.DEPTH_STENCIL_ATTACHMENT:e.DEPTH_ATTACHMENT,u=r.get(t),d=i.length>1;if(d)for(let t=0;t<i.length;t++)n.bindFramebuffer(e.FRAMEBUFFER,u.__webglMultisampledFramebuffer),e.framebufferRenderbuffer(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0+t,e.RENDERBUFFER,null),n.bindFramebuffer(e.FRAMEBUFFER,u.__webglFramebuffer),e.framebufferTexture2D(e.DRAW_FRAMEBUFFER,e.COLOR_ATTACHMENT0+t,e.TEXTURE_2D,null,0);n.bindFramebuffer(e.READ_FRAMEBUFFER,u.__webglMultisampledFramebuffer);let f=t.texture.mipmaps;f&&f.length>0?n.bindFramebuffer(e.DRAW_FRAMEBUFFER,u.__webglFramebuffer[0]):n.bindFramebuffer(e.DRAW_FRAMEBUFFER,u.__webglFramebuffer);for(let n=0;n<i.length;n++){if(t.resolveDepthBuffer&&(t.depthBuffer&&(s|=e.DEPTH_BUFFER_BIT),t.stencilBuffer&&t.resolveStencilBuffer&&(s|=e.STENCIL_BUFFER_BIT)),d){e.framebufferRenderbuffer(e.READ_FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.RENDERBUFFER,u.__webglColorRenderbuffer[n]);let t=r.get(i[n]).__webglTexture;e.framebufferTexture2D(e.DRAW_FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,t,0)}e.blitFramebuffer(0,0,a,o,0,0,a,o,s,e.NEAREST),c===!0&&(Se.length=0,Ce.length=0,Se.push(e.COLOR_ATTACHMENT0+n),t.depthBuffer&&t.resolveDepthBuffer===!1&&(Se.push(l),Ce.push(l),e.invalidateFramebuffer(e.DRAW_FRAMEBUFFER,Ce)),e.invalidateFramebuffer(e.READ_FRAMEBUFFER,Se))}if(n.bindFramebuffer(e.READ_FRAMEBUFFER,null),n.bindFramebuffer(e.DRAW_FRAMEBUFFER,null),d)for(let t=0;t<i.length;t++){n.bindFramebuffer(e.FRAMEBUFFER,u.__webglMultisampledFramebuffer),e.framebufferRenderbuffer(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0+t,e.RENDERBUFFER,u.__webglColorRenderbuffer[t]);let a=r.get(i[t]).__webglTexture;n.bindFramebuffer(e.FRAMEBUFFER,u.__webglFramebuffer),e.framebufferTexture2D(e.DRAW_FRAMEBUFFER,e.COLOR_ATTACHMENT0+t,e.TEXTURE_2D,a,0)}n.bindFramebuffer(e.DRAW_FRAMEBUFFER,u.__webglMultisampledFramebuffer)}else if(t.depthBuffer&&t.resolveDepthBuffer===!1&&c){let n=t.stencilBuffer?e.DEPTH_STENCIL_ATTACHMENT:e.DEPTH_ATTACHMENT;e.invalidateFramebuffer(e.DRAW_FRAMEBUFFER,[n])}}}function Te(e){return Math.min(i.maxSamples,e.samples)}function Ee(e){let n=r.get(e);return e.samples>0&&t.has(`WEBGL_multisampled_render_to_texture`)===!0&&n.__useRenderToTexture!==!1}function H(e){let t=o.render.frame;u.get(e)!==t&&(u.set(e,t),e.update())}function De(e,t){let n=e.colorSpace,r=e.format,i=e.type;return e.isCompressedTexture===!0||e.isVideoTexture===!0||n!==`srgb-linear`&&n!==``&&(Y.getTransfer(n)===`srgb`?(r!==1023||i!==1009)&&W(`WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType.`):G(`WebGLTextures: Unsupported texture color space:`,n)),t}function Oe(e){return typeof HTMLImageElement<`u`&&e instanceof HTMLImageElement?(l.width=e.naturalWidth||e.width,l.height=e.naturalHeight||e.height):typeof VideoFrame<`u`&&e instanceof VideoFrame?(l.width=e.displayWidth,l.height=e.displayHeight):(l.width=e.width,l.height=e.height),l}this.allocateTextureUnit=ne,this.resetTextureUnits=te,this.getTextureUnits=L,this.setTextureUnits=R,this.setTexture2D=ie,this.setTexture2DArray=z,this.setTexture3D=ae,this.setTextureCube=oe,this.rebindTextures=ye,this.setupRenderTarget=be,this.updateRenderTargetMipmap=xe,this.updateMultisampleRenderTarget=we,this.setupDepthRenderbuffer=ve,this.setupFrameBufferTexture=he,this.useMultisampledRTT=Ee,this.isReversedDepthBuffer=function(){return n.buffers.depth.getReversed()}}function ll(e,t){function n(n,r=``){let i,a=Y.getTransfer(r);if(n===1009)return e.UNSIGNED_BYTE;if(n===1017)return e.UNSIGNED_SHORT_4_4_4_4;if(n===1018)return e.UNSIGNED_SHORT_5_5_5_1;if(n===35902)return e.UNSIGNED_INT_5_9_9_9_REV;if(n===35899)return e.UNSIGNED_INT_10F_11F_11F_REV;if(n===1010)return e.BYTE;if(n===1011)return e.SHORT;if(n===1012)return e.UNSIGNED_SHORT;if(n===1013)return e.INT;if(n===1014)return e.UNSIGNED_INT;if(n===1015)return e.FLOAT;if(n===1016)return e.HALF_FLOAT;if(n===1021)return e.ALPHA;if(n===1022)return e.RGB;if(n===1023)return e.RGBA;if(n===1026)return e.DEPTH_COMPONENT;if(n===1027)return e.DEPTH_STENCIL;if(n===1028)return e.RED;if(n===1029)return e.RED_INTEGER;if(n===1030)return e.RG;if(n===1031)return e.RG_INTEGER;if(n===1033)return e.RGBA_INTEGER;if(n===33776||n===33777||n===33778||n===33779)if(a===`srgb`)if(i=t.get(`WEBGL_compressed_texture_s3tc_srgb`),i!==null){if(n===33776)return i.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===33777)return i.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===33778)return i.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===33779)return i.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(i=t.get(`WEBGL_compressed_texture_s3tc`),i!==null){if(n===33776)return i.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===33777)return i.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===33778)return i.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===33779)return i.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===35840||n===35841||n===35842||n===35843)if(i=t.get(`WEBGL_compressed_texture_pvrtc`),i!==null){if(n===35840)return i.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===35841)return i.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===35842)return i.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===35843)return i.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===36196||n===37492||n===37496||n===37488||n===37489||n===37490||n===37491)if(i=t.get(`WEBGL_compressed_texture_etc`),i!==null){if(n===36196||n===37492)return a===`srgb`?i.COMPRESSED_SRGB8_ETC2:i.COMPRESSED_RGB8_ETC2;if(n===37496)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:i.COMPRESSED_RGBA8_ETC2_EAC;if(n===37488)return i.COMPRESSED_R11_EAC;if(n===37489)return i.COMPRESSED_SIGNED_R11_EAC;if(n===37490)return i.COMPRESSED_RG11_EAC;if(n===37491)return i.COMPRESSED_SIGNED_RG11_EAC}else return null;if(n===37808||n===37809||n===37810||n===37811||n===37812||n===37813||n===37814||n===37815||n===37816||n===37817||n===37818||n===37819||n===37820||n===37821)if(i=t.get(`WEBGL_compressed_texture_astc`),i!==null){if(n===37808)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:i.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===37809)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:i.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===37810)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:i.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===37811)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:i.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===37812)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:i.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===37813)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:i.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===37814)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:i.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===37815)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:i.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===37816)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:i.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===37817)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:i.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===37818)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:i.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===37819)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:i.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===37820)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:i.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===37821)return a===`srgb`?i.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:i.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===36492||n===36494||n===36495)if(i=t.get(`EXT_texture_compression_bptc`),i!==null){if(n===36492)return a===`srgb`?i.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:i.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===36494)return i.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===36495)return i.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===36283||n===36284||n===36285||n===36286)if(i=t.get(`EXT_texture_compression_rgtc`),i!==null){if(n===36283)return i.COMPRESSED_RED_RGTC1_EXT;if(n===36284)return i.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===36285)return i.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===36286)return i.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===1020?e.UNSIGNED_INT_24_8:e[n]===void 0?null:e[n]}return{convert:n}}var ul=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,dl=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`,fl=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){let n=new Gi(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){let t=e.cameras[0].viewport,n=new na({vertexShader:ul,fragmentShader:dl,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new Si(new qi(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},pl=class extends Ct{constructor(e,t){super();let n=this,r=null,i=1,a=null,o=`local-floor`,s=1,c=null,l=null,u=null,d=null,f=null,p=null,m=typeof XRWebGLBinding<`u`,h=new fl,g={},_=t.getContextAttributes(),v=null,y=null,b=[],x=[],S=new Mt,C=null,w=new Ia;w.viewport=new Xt;let T=new Ia;T.viewport=new Xt;let E=[w,T],D=new Ua,O=null,k=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(e){let t=b[e];return t===void 0&&(t=new Mn,b[e]=t),t.getTargetRaySpace()},this.getControllerGrip=function(e){let t=b[e];return t===void 0&&(t=new Mn,b[e]=t),t.getGripSpace()},this.getHand=function(e){let t=b[e];return t===void 0&&(t=new Mn,b[e]=t),t.getHandSpace()};function A(e){let t=x.indexOf(e.inputSource);if(t===-1)return;let n=b[t];n!==void 0&&(n.update(e.inputSource,e.frame,c||a),n.dispatchEvent({type:e.type,data:e.inputSource}))}function j(){r.removeEventListener(`select`,A),r.removeEventListener(`selectstart`,A),r.removeEventListener(`selectend`,A),r.removeEventListener(`squeeze`,A),r.removeEventListener(`squeezestart`,A),r.removeEventListener(`squeezeend`,A),r.removeEventListener(`end`,j),r.removeEventListener(`inputsourceschange`,M);for(let e=0;e<b.length;e++){let t=x[e];t!==null&&(x[e]=null,b[e].disconnect(t))}O=null,k=null,h.reset();for(let e in g)delete g[e];e.setRenderTarget(v),f=null,d=null,u=null,r=null,y=null,re.stop(),n.isPresenting=!1,e.setPixelRatio(C),e.setSize(S.width,S.height,!1),n.dispatchEvent({type:`sessionend`})}this.setFramebufferScaleFactor=function(e){i=e,n.isPresenting===!0&&W(`WebXRManager: Cannot change framebuffer scale while presenting.`)},this.setReferenceSpaceType=function(e){o=e,n.isPresenting===!0&&W(`WebXRManager: Cannot change reference space type while presenting.`)},this.getReferenceSpace=function(){return c||a},this.setReferenceSpace=function(e){c=e},this.getBaseLayer=function(){return d===null?f:d},this.getBinding=function(){return u===null&&m&&(u=new XRWebGLBinding(r,t)),u},this.getFrame=function(){return p},this.getSession=function(){return r},this.setSession=async function(l){if(r=l,r!==null){if(v=e.getRenderTarget(),r.addEventListener(`select`,A),r.addEventListener(`selectstart`,A),r.addEventListener(`selectend`,A),r.addEventListener(`squeeze`,A),r.addEventListener(`squeezestart`,A),r.addEventListener(`squeezeend`,A),r.addEventListener(`end`,j),r.addEventListener(`inputsourceschange`,M),_.xrCompatible!==!0&&await t.makeXRCompatible(),C=e.getPixelRatio(),e.getSize(S),m&&`createProjectionLayer`in XRWebGLBinding.prototype){let n=null,a=null,o=null;_.depth&&(o=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,n=_.stencil?de:V,a=_.stencil?oe:ne);let s={colorFormat:t.RGBA8,depthFormat:o,scaleFactor:i};u=this.getBinding(),d=u.createProjectionLayer(s),r.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),y=new Qt(d.textureWidth,d.textureHeight,{format:ue,type:I,depthTexture:new Ui(d.textureWidth,d.textureHeight,a,void 0,void 0,void 0,void 0,void 0,void 0,n),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{let n={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:i};f=new XRWebGLLayer(r,t,n),r.updateRenderState({baseLayer:f}),e.setPixelRatio(1),e.setSize(f.framebufferWidth,f.framebufferHeight,!1),y=new Qt(f.framebufferWidth,f.framebufferHeight,{format:ue,type:I,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}y.isXRRenderTarget=!0,this.setFoveation(s),c=null,a=await r.requestReferenceSpace(o),re.setContext(r),re.start(),n.isPresenting=!0,n.dispatchEvent({type:`sessionstart`})}},this.getEnvironmentBlendMode=function(){if(r!==null)return r.environmentBlendMode},this.getDepthTexture=function(){return h.getDepthTexture()};function M(e){for(let t=0;t<e.removed.length;t++){let n=e.removed[t],r=x.indexOf(n);r>=0&&(x[r]=null,b[r].disconnect(n))}for(let t=0;t<e.added.length;t++){let n=e.added[t],r=x.indexOf(n);if(r===-1){for(let e=0;e<b.length;e++)if(e>=x.length){x.push(n),r=e;break}else if(x[e]===null){x[e]=n,r=e;break}if(r===-1)break}let i=b[r];i&&i.connect(n)}}let N=new q,P=new q;function F(e,t,n){N.setFromMatrixPosition(t.matrixWorld),P.setFromMatrixPosition(n.matrixWorld);let r=N.distanceTo(P),i=t.projectionMatrix.elements,a=n.projectionMatrix.elements,o=i[14]/(i[10]-1),s=i[14]/(i[10]+1),c=(i[9]+1)/i[5],l=(i[9]-1)/i[5],u=(i[8]-1)/i[0],d=(a[8]+1)/a[0],f=o*u,p=o*d,m=r/(-u+d),h=m*-u;if(t.matrixWorld.decompose(e.position,e.quaternion,e.scale),e.translateX(h),e.translateZ(m),e.matrixWorld.compose(e.position,e.quaternion,e.scale),e.matrixWorldInverse.copy(e.matrixWorld).invert(),i[10]===-1)e.projectionMatrix.copy(t.projectionMatrix),e.projectionMatrixInverse.copy(t.projectionMatrixInverse);else{let t=o+m,n=s+m,i=f-h,a=p+(r-h),u=c*s/n*t,d=l*s/n*t;e.projectionMatrix.makePerspective(i,a,u,d,t,n),e.projectionMatrixInverse.copy(e.projectionMatrix).invert()}}function ee(e,t){t===null?e.matrixWorld.copy(e.matrix):e.matrixWorld.multiplyMatrices(t.matrixWorld,e.matrix),e.matrixWorldInverse.copy(e.matrixWorld).invert()}this.updateCamera=function(e){if(r===null)return;let t=e.near,n=e.far;h.texture!==null&&(h.depthNear>0&&(t=h.depthNear),h.depthFar>0&&(n=h.depthFar)),D.near=T.near=w.near=t,D.far=T.far=w.far=n,(O!==D.near||k!==D.far)&&(r.updateRenderState({depthNear:D.near,depthFar:D.far}),O=D.near,k=D.far),D.layers.mask=e.layers.mask|6,w.layers.mask=D.layers.mask&-5,T.layers.mask=D.layers.mask&-3;let i=e.parent,a=D.cameras;ee(D,i);for(let e=0;e<a.length;e++)ee(a[e],i);a.length===2?F(D,w,T):D.projectionMatrix.copy(w.projectionMatrix),te(e,D,i)};function te(e,t,n){n===null?e.matrix.copy(t.matrixWorld):(e.matrix.copy(n.matrixWorld),e.matrix.invert(),e.matrix.multiply(t.matrixWorld)),e.matrix.decompose(e.position,e.quaternion,e.scale),e.updateMatrixWorld(!0),e.projectionMatrix.copy(t.projectionMatrix),e.projectionMatrixInverse.copy(t.projectionMatrixInverse),e.isPerspectiveCamera&&(e.fov=Et*2*Math.atan(1/e.projectionMatrix.elements[5]),e.zoom=1)}this.getCamera=function(){return D},this.getFoveation=function(){if(!(d===null&&f===null))return s},this.setFoveation=function(e){s=e,d!==null&&(d.fixedFoveation=e),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=e)},this.hasDepthSensing=function(){return h.texture!==null},this.getDepthSensingMesh=function(){return h.getMesh(D)},this.getCameraTexture=function(e){return g[e]};let L=null;function R(t,i){if(l=i.getViewerPose(c||a),p=i,l!==null){let t=l.views;f!==null&&(e.setRenderTargetFramebuffer(y,f.framebuffer),e.setRenderTarget(y));let i=!1;t.length!==D.cameras.length&&(D.cameras.length=0,i=!0);for(let n=0;n<t.length;n++){let r=t[n],a=null;if(f!==null)a=f.getViewport(r);else{let t=u.getViewSubImage(d,r);a=t.viewport,n===0&&(e.setRenderTargetTextures(y,t.colorTexture,t.depthStencilTexture),e.setRenderTarget(y))}let o=E[n];o===void 0&&(o=new Ia,o.layers.enable(n),o.viewport=new Xt,E[n]=o),o.matrix.fromArray(r.transform.matrix),o.matrix.decompose(o.position,o.quaternion,o.scale),o.projectionMatrix.fromArray(r.projectionMatrix),o.projectionMatrixInverse.copy(o.projectionMatrix).invert(),o.viewport.set(a.x,a.y,a.width,a.height),n===0&&(D.matrix.copy(o.matrix),D.matrix.decompose(D.position,D.quaternion,D.scale)),i===!0&&D.cameras.push(o)}let a=r.enabledFeatures;if(a&&a.includes(`depth-sensing`)&&r.depthUsage==`gpu-optimized`&&m){u=n.getBinding();let e=u.getDepthInformation(t[0]);e&&e.isValid&&e.texture&&h.init(e,r.renderState)}if(a&&a.includes(`camera-access`)&&m){e.state.unbindTexture(),u=n.getBinding();for(let e=0;e<t.length;e++){let n=t[e].camera;if(n){let e=g[n];e||(e=new Gi,g[n]=e);let t=u.getCameraImage(n);e.sourceTexture=t}}}}for(let e=0;e<b.length;e++){let t=x[e],n=b[e];t!==null&&n!==void 0&&n.update(t,i,c||a)}L&&L(t,i),i.detectedPlanes&&n.dispatchEvent({type:`planesdetected`,data:i}),p=null}let re=new io;re.setAnimationLoop(R),this.setAnimationLoop=function(e){L=e},this.dispose=function(){}}},ml=new tn,hl=new J;hl.set(-1,0,0,0,1,0,0,0,1);function gl(e,t){function n(e,t){e.matrixAutoUpdate===!0&&e.updateMatrix(),t.value.copy(e.matrix)}function r(t,n){n.color.getRGB(t.fogColor.value,Qi(e)),n.isFog?(t.fogNear.value=n.near,t.fogFar.value=n.far):n.isFogExp2&&(t.fogDensity.value=n.density)}function i(e,t,n,r,i){t.isNodeMaterial?t.uniformsNeedUpdate=!1:t.isMeshBasicMaterial?a(e,t):t.isMeshLambertMaterial?(a(e,t),t.envMap&&(e.envMapIntensity.value=t.envMapIntensity)):t.isMeshToonMaterial?(a(e,t),d(e,t)):t.isMeshPhongMaterial?(a(e,t),u(e,t),t.envMap&&(e.envMapIntensity.value=t.envMapIntensity)):t.isMeshStandardMaterial?(a(e,t),f(e,t),t.isMeshPhysicalMaterial&&p(e,t,i)):t.isMeshMatcapMaterial?(a(e,t),m(e,t)):t.isMeshDepthMaterial?a(e,t):t.isMeshDistanceMaterial?(a(e,t),h(e,t)):t.isMeshNormalMaterial?a(e,t):t.isLineBasicMaterial?(o(e,t),t.isLineDashedMaterial&&s(e,t)):t.isPointsMaterial?c(e,t,n,r):t.isSpriteMaterial?l(e,t):t.isShadowMaterial?(e.color.value.copy(t.color),e.opacity.value=t.opacity):t.isShaderMaterial&&(t.uniformsNeedUpdate=!1)}function a(e,r){e.opacity.value=r.opacity,r.color&&e.diffuse.value.copy(r.color),r.emissive&&e.emissive.value.copy(r.emissive).multiplyScalar(r.emissiveIntensity),r.map&&(e.map.value=r.map,n(r.map,e.mapTransform)),r.alphaMap&&(e.alphaMap.value=r.alphaMap,n(r.alphaMap,e.alphaMapTransform)),r.bumpMap&&(e.bumpMap.value=r.bumpMap,n(r.bumpMap,e.bumpMapTransform),e.bumpScale.value=r.bumpScale,r.side===1&&(e.bumpScale.value*=-1)),r.normalMap&&(e.normalMap.value=r.normalMap,n(r.normalMap,e.normalMapTransform),e.normalScale.value.copy(r.normalScale),r.side===1&&e.normalScale.value.negate()),r.displacementMap&&(e.displacementMap.value=r.displacementMap,n(r.displacementMap,e.displacementMapTransform),e.displacementScale.value=r.displacementScale,e.displacementBias.value=r.displacementBias),r.emissiveMap&&(e.emissiveMap.value=r.emissiveMap,n(r.emissiveMap,e.emissiveMapTransform)),r.specularMap&&(e.specularMap.value=r.specularMap,n(r.specularMap,e.specularMapTransform)),r.alphaTest>0&&(e.alphaTest.value=r.alphaTest);let i=t.get(r),a=i.envMap,o=i.envMapRotation;a&&(e.envMap.value=a,e.envMapRotation.value.setFromMatrix4(ml.makeRotationFromEuler(o)).transpose(),a.isCubeTexture&&a.isRenderTargetTexture===!1&&e.envMapRotation.value.premultiply(hl),e.reflectivity.value=r.reflectivity,e.ior.value=r.ior,e.refractionRatio.value=r.refractionRatio),r.lightMap&&(e.lightMap.value=r.lightMap,e.lightMapIntensity.value=r.lightMapIntensity,n(r.lightMap,e.lightMapTransform)),r.aoMap&&(e.aoMap.value=r.aoMap,e.aoMapIntensity.value=r.aoMapIntensity,n(r.aoMap,e.aoMapTransform))}function o(e,t){e.diffuse.value.copy(t.color),e.opacity.value=t.opacity,t.map&&(e.map.value=t.map,n(t.map,e.mapTransform))}function s(e,t){e.dashSize.value=t.dashSize,e.totalSize.value=t.dashSize+t.gapSize,e.scale.value=t.scale}function c(e,t,r,i){e.diffuse.value.copy(t.color),e.opacity.value=t.opacity,e.size.value=t.size*r,e.scale.value=i*.5,t.map&&(e.map.value=t.map,n(t.map,e.uvTransform)),t.alphaMap&&(e.alphaMap.value=t.alphaMap,n(t.alphaMap,e.alphaMapTransform)),t.alphaTest>0&&(e.alphaTest.value=t.alphaTest)}function l(e,t){e.diffuse.value.copy(t.color),e.opacity.value=t.opacity,e.rotation.value=t.rotation,t.map&&(e.map.value=t.map,n(t.map,e.mapTransform)),t.alphaMap&&(e.alphaMap.value=t.alphaMap,n(t.alphaMap,e.alphaMapTransform)),t.alphaTest>0&&(e.alphaTest.value=t.alphaTest)}function u(e,t){e.specular.value.copy(t.specular),e.shininess.value=Math.max(t.shininess,1e-4)}function d(e,t){t.gradientMap&&(e.gradientMap.value=t.gradientMap)}function f(e,t){e.metalness.value=t.metalness,t.metalnessMap&&(e.metalnessMap.value=t.metalnessMap,n(t.metalnessMap,e.metalnessMapTransform)),e.roughness.value=t.roughness,t.roughnessMap&&(e.roughnessMap.value=t.roughnessMap,n(t.roughnessMap,e.roughnessMapTransform)),t.envMap&&(e.envMapIntensity.value=t.envMapIntensity)}function p(e,t,r){e.ior.value=t.ior,t.sheen>0&&(e.sheenColor.value.copy(t.sheenColor).multiplyScalar(t.sheen),e.sheenRoughness.value=t.sheenRoughness,t.sheenColorMap&&(e.sheenColorMap.value=t.sheenColorMap,n(t.sheenColorMap,e.sheenColorMapTransform)),t.sheenRoughnessMap&&(e.sheenRoughnessMap.value=t.sheenRoughnessMap,n(t.sheenRoughnessMap,e.sheenRoughnessMapTransform))),t.clearcoat>0&&(e.clearcoat.value=t.clearcoat,e.clearcoatRoughness.value=t.clearcoatRoughness,t.clearcoatMap&&(e.clearcoatMap.value=t.clearcoatMap,n(t.clearcoatMap,e.clearcoatMapTransform)),t.clearcoatRoughnessMap&&(e.clearcoatRoughnessMap.value=t.clearcoatRoughnessMap,n(t.clearcoatRoughnessMap,e.clearcoatRoughnessMapTransform)),t.clearcoatNormalMap&&(e.clearcoatNormalMap.value=t.clearcoatNormalMap,n(t.clearcoatNormalMap,e.clearcoatNormalMapTransform),e.clearcoatNormalScale.value.copy(t.clearcoatNormalScale),t.side===1&&e.clearcoatNormalScale.value.negate())),t.dispersion>0&&(e.dispersion.value=t.dispersion),t.iridescence>0&&(e.iridescence.value=t.iridescence,e.iridescenceIOR.value=t.iridescenceIOR,e.iridescenceThicknessMinimum.value=t.iridescenceThicknessRange[0],e.iridescenceThicknessMaximum.value=t.iridescenceThicknessRange[1],t.iridescenceMap&&(e.iridescenceMap.value=t.iridescenceMap,n(t.iridescenceMap,e.iridescenceMapTransform)),t.iridescenceThicknessMap&&(e.iridescenceThicknessMap.value=t.iridescenceThicknessMap,n(t.iridescenceThicknessMap,e.iridescenceThicknessMapTransform))),t.transmission>0&&(e.transmission.value=t.transmission,e.transmissionSamplerMap.value=r.texture,e.transmissionSamplerSize.value.set(r.width,r.height),t.transmissionMap&&(e.transmissionMap.value=t.transmissionMap,n(t.transmissionMap,e.transmissionMapTransform)),e.thickness.value=t.thickness,t.thicknessMap&&(e.thicknessMap.value=t.thicknessMap,n(t.thicknessMap,e.thicknessMapTransform)),e.attenuationDistance.value=t.attenuationDistance,e.attenuationColor.value.copy(t.attenuationColor)),t.anisotropy>0&&(e.anisotropyVector.value.set(t.anisotropy*Math.cos(t.anisotropyRotation),t.anisotropy*Math.sin(t.anisotropyRotation)),t.anisotropyMap&&(e.anisotropyMap.value=t.anisotropyMap,n(t.anisotropyMap,e.anisotropyMapTransform))),e.specularIntensity.value=t.specularIntensity,e.specularColor.value.copy(t.specularColor),t.specularColorMap&&(e.specularColorMap.value=t.specularColorMap,n(t.specularColorMap,e.specularColorMapTransform)),t.specularIntensityMap&&(e.specularIntensityMap.value=t.specularIntensityMap,n(t.specularIntensityMap,e.specularIntensityMapTransform))}function m(e,t){t.matcap&&(e.matcap.value=t.matcap)}function h(e,n){let r=t.get(n).light;e.referencePosition.value.setFromMatrixPosition(r.matrixWorld),e.nearDistance.value=r.shadow.camera.near,e.farDistance.value=r.shadow.camera.far}return{refreshFogUniforms:r,refreshMaterialUniforms:i}}function _l(e,t,n,r){let i={},a={},o=[],s=e.getParameter(e.MAX_UNIFORM_BUFFER_BINDINGS);function c(e,t){let n=t.program;r.uniformBlockBinding(e,n)}function l(e,n){let o=i[e.id];o===void 0&&(g(e),o=u(e),i[e.id]=o,e.addEventListener(`dispose`,v));let s=n.program;r.updateUBOMapping(e,s);let c=t.render.frame;a[e.id]!==c&&(f(e),a[e.id]=c)}function u(t){let n=d();t.__bindingPointIndex=n;let r=e.createBuffer(),i=t.__size,a=t.usage;return e.bindBuffer(e.UNIFORM_BUFFER,r),e.bufferData(e.UNIFORM_BUFFER,i,a),e.bindBuffer(e.UNIFORM_BUFFER,null),e.bindBufferBase(e.UNIFORM_BUFFER,n,r),r}function d(){for(let e=0;e<s;e++)if(o.indexOf(e)===-1)return o.push(e),e;return G(`WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached.`),0}function f(t){let n=i[t.id],r=t.uniforms,a=t.__cache;e.bindBuffer(e.UNIFORM_BUFFER,n);for(let e=0,t=r.length;e<t;e++){let t=r[e];if(Array.isArray(t))for(let n=0,r=t.length;n<r;n++)p(t[n],e,n,a);else p(t,e,0,a)}e.bindBuffer(e.UNIFORM_BUFFER,null)}function p(t,n,r,i){if(h(t,n,r,i)===!0){let n=t.__offset,r=t.value;if(Array.isArray(r)){let e=0;for(let n=0;n<r.length;n++){let i=r[n],a=_(i);m(i,t.__data,e),typeof i!=`number`&&typeof i!=`boolean`&&!i.isMatrix3&&!ArrayBuffer.isView(i)&&(e+=a.storage/Float32Array.BYTES_PER_ELEMENT)}}else m(r,t.__data,0);e.bufferSubData(e.UNIFORM_BUFFER,n,t.__data)}}function m(e,t,n){typeof e==`number`||typeof e==`boolean`?t[0]=e:e.isMatrix3?(t[0]=e.elements[0],t[1]=e.elements[1],t[2]=e.elements[2],t[3]=0,t[4]=e.elements[3],t[5]=e.elements[4],t[6]=e.elements[5],t[7]=0,t[8]=e.elements[6],t[9]=e.elements[7],t[10]=e.elements[8],t[11]=0):ArrayBuffer.isView(e)?t.set(new e.constructor(e.buffer,e.byteOffset,t.length)):e.toArray(t,n)}function h(e,t,n,r){let i=e.value,a=t+`_`+n;if(r[a]===void 0)return typeof i==`number`||typeof i==`boolean`?r[a]=i:ArrayBuffer.isView(i)?r[a]=i.slice():r[a]=i.clone(),!0;{let e=r[a];if(typeof i==`number`||typeof i==`boolean`){if(e!==i)return r[a]=i,!0}else if(ArrayBuffer.isView(i))return!0;else if(e.equals(i)===!1)return e.copy(i),!0}return!1}function g(e){let t=e.uniforms,n=0;for(let e=0,r=t.length;e<r;e++){let r=Array.isArray(t[e])?t[e]:[t[e]];for(let e=0,t=r.length;e<t;e++){let t=r[e],i=Array.isArray(t.value)?t.value:[t.value];for(let e=0,r=i.length;e<r;e++){let r=i[e],a=_(r),o=n%16,s=o%a.boundary,c=o+s;n+=s,c!==0&&16-c<a.storage&&(n+=16-c),t.__data=new Float32Array(a.storage/Float32Array.BYTES_PER_ELEMENT),t.__offset=n,n+=a.storage}}}let r=n%16;return r>0&&(n+=16-r),e.__size=n,e.__cache={},this}function _(e){let t={boundary:0,storage:0};return typeof e==`number`||typeof e==`boolean`?(t.boundary=4,t.storage=4):e.isVector2?(t.boundary=8,t.storage=8):e.isVector3||e.isColor?(t.boundary=16,t.storage=12):e.isVector4?(t.boundary=16,t.storage=16):e.isMatrix3?(t.boundary=48,t.storage=48):e.isMatrix4?(t.boundary=64,t.storage=64):e.isTexture?W(`WebGLRenderer: Texture samplers can not be part of an uniforms group.`):ArrayBuffer.isView(e)?(t.boundary=16,t.storage=e.byteLength):W(`WebGLRenderer: Unsupported uniform value type.`,e),t}function v(t){let n=t.target;n.removeEventListener(`dispose`,v);let r=o.indexOf(n.__bindingPointIndex);o.splice(r,1),e.deleteBuffer(i[n.id]),delete i[n.id],delete a[n.id]}function y(){for(let t in i)e.deleteBuffer(i[t]);o=[],i={},a={}}return{bind:c,update:l,dispose:y}}var vl=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),yl=null;function bl(){return yl===null&&(yl=new Ti(vl,16,16,me,ie),yl.name=`DFG_LUT`,yl.minFilter=N,yl.magFilter=N,yl.wrapS=O,yl.wrapT=O,yl.generateMipmaps=!1,yl.needsUpdate=!0),yl}var xl=class{constructor(e={}){let{canvas:t=ht(),context:n=null,depth:r=!0,stencil:i=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:s=!0,preserveDrawingBuffer:c=!1,powerPreference:l=`default`,failIfMajorPerformanceCaveat:u=!1,reversedDepthBuffer:d=!1,outputBufferType:f=I}=e;this.isWebGLRenderer=!0;let p;if(n!==null){if(typeof WebGLRenderingContext<`u`&&n instanceof WebGLRenderingContext)throw Error(`THREE.WebGLRenderer: WebGL 1 is not supported since r163.`);p=n.getContextAttributes().alpha}else p=a;let m=f,h=new Set([ge,he,pe]),g=new Set([I,ne,L,oe,z,ae]),_=new Uint32Array(4),v=new Int32Array(4),y=new q,b=null,x=null,S=[],C=[],w=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=0,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let T=this,E=!1,D=null,O=null,k=null,A=null;this._outputColorSpace=at;let j=0,M=0,N=null,P=-1,ee=null,te=new Xt,R=new Xt,re=null,se=new Ln(0),ce=0,le=t.width,B=t.height,ue=1,V=null,de=null,fe=new Xt(0,0,le,B),me=new Xt(0,0,le,B),_e=!1,ve=new Ni,ye=!1,be=!1,xe=new tn,Se=new q,Ce=new Xt,we={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},Te=!1;function Ee(){return N===null?ue:1}let H=n;function De(e,n){return t.getContext(e,n)}try{let e={alpha:!0,depth:r,stencil:i,antialias:o,premultipliedAlpha:s,preserveDrawingBuffer:c,powerPreference:l,failIfMajorPerformanceCaveat:u};if(`setAttribute`in t&&t.setAttribute(`data-engine`,`three.js r185`),t.addEventListener(`webglcontextlost`,Qe,!1),t.addEventListener(`webglcontextrestored`,$e,!1),t.addEventListener(`webglcontextcreationerror`,et,!1),H===null){let t=`webgl2`;if(H=De(t,e),H===null)throw De(t)?Error(`THREE.WebGLRenderer: Error creating WebGL context with your selected attributes.`):Error(`THREE.WebGLRenderer: Error creating WebGL context.`)}}catch(e){throw G(`WebGLRenderer: `+e.message),e}let Oe,ke,U,Ae,je,Me,Ne,Pe,Fe,Ie,Le,Re,ze,Be,Ve,He,Ue,We,Ge,Ke,qe,Je,Ye;function Xe(){Oe=new zo(H),Oe.init(),qe=new ll(H,Oe),ke=new ho(H,Oe,e,qe),U=new sl(H,Oe),ke.reversedDepthBuffer&&d&&U.buffers.depth.setReversed(!0),O=H.createFramebuffer(),k=H.createFramebuffer(),A=H.createFramebuffer(),Ae=new Ho(H),je=new Vc,Me=new cl(H,Oe,U,je,ke,qe,Ae),Ne=new Ro(T),Pe=new ao(H),Je=new po(H,Pe),Fe=new Bo(H,Pe,Ae,Je),Ie=new Wo(H,Fe,Pe,Je,Ae),We=new Uo(H,ke,Me),Ve=new go(je),Le=new Bc(T,Ne,Oe,ke,Je,Ve),Re=new gl(T,je),ze=new Gc,Be=new Qc(Oe),Ue=new fo(T,Ne,U,Ie,p,s),He=new ol(T,Ie,ke),Ye=new _l(H,Ae,ke,U),Ge=new mo(H,Oe,Ae),Ke=new Vo(H,Oe,Ae),Ae.programs=Le.programs,T.capabilities=ke,T.extensions=Oe,T.properties=je,T.renderLists=ze,T.shadowMap=He,T.state=U,T.info=Ae}Xe(),m!==1009&&(w=new Ko(m,t.width,t.height,o,r,i));let Ze=new pl(T,H);this.xr=Ze,this.getContext=function(){return H},this.getContextAttributes=function(){return H.getContextAttributes()},this.forceContextLoss=function(){let e=Oe.get(`WEBGL_lose_context`);e&&e.loseContext()},this.forceContextRestore=function(){let e=Oe.get(`WEBGL_lose_context`);e&&e.restoreContext()},this.getPixelRatio=function(){return ue},this.setPixelRatio=function(e){e!==void 0&&(ue=e,this.setSize(le,B,!1))},this.getSize=function(e){return e.set(le,B)},this.setSize=function(e,n,r=!0){if(Ze.isPresenting){W(`WebGLRenderer: Can't change size while VR device is presenting.`);return}le=e,B=n,t.width=Math.floor(e*ue),t.height=Math.floor(n*ue),r===!0&&(t.style.width=e+`px`,t.style.height=n+`px`),w!==null&&w.setSize(t.width,t.height),this.setViewport(0,0,e,n)},this.getDrawingBufferSize=function(e){return e.set(le*ue,B*ue).floor()},this.setDrawingBufferSize=function(e,n,r){le=e,B=n,ue=r,t.width=Math.floor(e*r),t.height=Math.floor(n*r),this.setViewport(0,0,e,n)},this.setEffects=function(e){if(m===1009){G(`WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.`);return}if(e){for(let t=0;t<e.length;t++)if(e[t].isOutputPass===!0){W(`WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.`);break}}w.setEffects(e||[])},this.getCurrentViewport=function(e){return e.copy(te)},this.getViewport=function(e){return e.copy(fe)},this.setViewport=function(e,t,n,r){e.isVector4?fe.set(e.x,e.y,e.z,e.w):fe.set(e,t,n,r),U.viewport(te.copy(fe).multiplyScalar(ue).round())},this.getScissor=function(e){return e.copy(me)},this.setScissor=function(e,t,n,r){e.isVector4?me.set(e.x,e.y,e.z,e.w):me.set(e,t,n,r),U.scissor(R.copy(me).multiplyScalar(ue).round())},this.getScissorTest=function(){return _e},this.setScissorTest=function(e){U.setScissorTest(_e=e)},this.setOpaqueSort=function(e){V=e},this.setTransparentSort=function(e){de=e},this.getClearColor=function(e){return e.copy(Ue.getClearColor())},this.setClearColor=function(){Ue.setClearColor(...arguments)},this.getClearAlpha=function(){return Ue.getClearAlpha()},this.setClearAlpha=function(){Ue.setClearAlpha(...arguments)},this.clear=function(e=!0,t=!0,n=!0){let r=0;if(e){let e=!1;if(N!==null){let t=N.texture.format;e=h.has(t)}if(e){let e=N.texture.type,t=g.has(e),n=Ue.getClearColor(),r=Ue.getClearAlpha(),i=n.r,a=n.g,o=n.b;t?(_[0]=i,_[1]=a,_[2]=o,_[3]=r,H.clearBufferuiv(H.COLOR,0,_)):(v[0]=i,v[1]=a,v[2]=o,v[3]=r,H.clearBufferiv(H.COLOR,0,v))}else r|=H.COLOR_BUFFER_BIT}t&&(r|=H.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),n&&(r|=H.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),r!==0&&H.clear(r)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(e){e.setRenderer(this),D=e},this.dispose=function(){t.removeEventListener(`webglcontextlost`,Qe,!1),t.removeEventListener(`webglcontextrestored`,$e,!1),t.removeEventListener(`webglcontextcreationerror`,et,!1),Ue.dispose(),ze.dispose(),Be.dispose(),je.dispose(),Ne.dispose(),Ie.dispose(),Je.dispose(),Ye.dispose(),Le.dispose(),Ze.dispose(),Ze.removeEventListener(`sessionstart`,ct),Ze.removeEventListener(`sessionend`,lt),ut.stop()};function Qe(e){e.preventDefault(),vt(`WebGLRenderer: Context Lost.`),E=!0}function $e(){vt(`WebGLRenderer: Context Restored.`),E=!1;let e=Ae.autoReset,t=He.enabled,n=He.autoUpdate,r=He.needsUpdate,i=He.type;Xe(),Ae.autoReset=e,He.enabled=t,He.autoUpdate=n,He.needsUpdate=r,He.type=i}function et(e){G(`WebGLRenderer: A WebGL context could not be created. Reason: `,e.statusMessage)}function tt(e){let t=e.target;t.removeEventListener(`dispose`,tt),nt(t)}function nt(e){rt(e),je.remove(e)}function rt(e){let t=je.get(e).programs;t!==void 0&&(t.forEach(function(e){Le.releaseProgram(e)}),e.isShaderMaterial&&Le.releaseShaderCache(e))}this.renderBufferDirect=function(e,t,n,r,i,a){t===null&&(t=we);let o=i.isMesh&&i.matrixWorld.determinantAffine()<0,s=wt(e,t,n,r,i);U.setMaterial(r,o);let c=n.index,l=1;if(r.wireframe===!0){if(c=Fe.getWireframeAttribute(n),c===void 0)return;l=2}let u=n.drawRange,d=n.attributes.position,f=u.start*l,p=(u.start+u.count)*l;a!==null&&(f=Math.max(f,a.start*l),p=Math.min(p,(a.start+a.count)*l)),c===null?d!=null&&(f=Math.max(f,0),p=Math.min(p,d.count)):(f=Math.max(f,0),p=Math.min(p,c.count));let m=p-f;if(m<0||m===1/0)return;Je.setup(i,r,s,n,c);let h,g=Ge;if(c!==null&&(h=Pe.get(c),g=Ke,g.setIndex(h)),i.isMesh)r.wireframe===!0?(U.setLineWidth(r.wireframeLinewidth*Ee()),g.setMode(H.LINES)):g.setMode(H.TRIANGLES);else if(i.isLine){let e=r.linewidth;e===void 0&&(e=1),U.setLineWidth(e*Ee()),i.isLineSegments?g.setMode(H.LINES):i.isLineLoop?g.setMode(H.LINE_LOOP):g.setMode(H.LINE_STRIP)}else i.isPoints?g.setMode(H.POINTS):i.isSprite&&g.setMode(H.TRIANGLES);if(i.isBatchedMesh)if(Oe.get(`WEBGL_multi_draw`))g.renderMultiDraw(i._multiDrawStarts,i._multiDrawCounts,i._multiDrawCount);else{let e=i._multiDrawStarts,t=i._multiDrawCounts,n=i._multiDrawCount,a=c?Pe.get(c).bytesPerElement:1,o=je.get(r).currentProgram.getUniforms();for(let r=0;r<n;r++)o.setValue(H,`_gl_DrawID`,r),g.render(e[r]/a,t[r])}else if(i.isInstancedMesh)g.renderInstances(f,m,i.count);else if(n.isInstancedBufferGeometry){let e=n._maxInstanceCount===void 0?1/0:n._maxInstanceCount,t=Math.min(n.instanceCount,e);g.renderInstances(f,m,t)}else g.render(f,m)};function it(e,t,n){e.transparent===!0&&e.side===2&&e.forceSinglePass===!1?(e.side=1,e.needsUpdate=!0,yt(e,t,n),e.side=0,e.needsUpdate=!0,yt(e,t,n),e.side=2):yt(e,t,n)}this.compile=function(e,t,n=null){n===null&&(n=e),x=Be.get(n),x.init(t),C.push(x),n.traverseVisible(function(e){e.isLight&&e.layers.test(t.layers)&&(x.pushLight(e),e.castShadow&&x.pushShadow(e))}),e!==n&&e.traverseVisible(function(e){e.isLight&&e.layers.test(t.layers)&&(x.pushLight(e),e.castShadow&&x.pushShadow(e))}),x.setupLights();let r=new Set;return e.traverse(function(e){if(!(e.isMesh||e.isPoints||e.isLine||e.isSprite))return;let t=e.material;if(t)if(Array.isArray(t))for(let i=0;i<t.length;i++){let a=t[i];it(a,n,e),r.add(a)}else it(t,n,e),r.add(t)}),x=C.pop(),r},this.compileAsync=function(e,t,n=null){let r=this.compile(e,t,n);return new Promise(t=>{function n(){if(r.forEach(function(e){je.get(e).currentProgram.isReady()&&r.delete(e)}),r.size===0){t(e);return}setTimeout(n,10)}Oe.get(`KHR_parallel_shader_compile`)===null?setTimeout(n,10):n()})};let ot=null;function st(e){ot&&ot(e)}function ct(){ut.stop()}function lt(){ut.start()}let ut=new io;ut.setAnimationLoop(st),typeof self<`u`&&ut.setContext(self),this.setAnimationLoop=function(e){ot=e,Ze.setAnimationLoop(e),e===null?ut.stop():ut.start()},Ze.addEventListener(`sessionstart`,ct),Ze.addEventListener(`sessionend`,lt),this.render=function(e,t){if(t!==void 0&&t.isCamera!==!0){G(`WebGLRenderer.render: camera is not an instance of THREE.Camera.`);return}if(E===!0)return;D!==null&&D.renderStart(e,t);let n=Ze.enabled===!0&&Ze.isPresenting===!0,r=w!==null&&(N===null||n)&&w.begin(T,N);if(e.matrixWorldAutoUpdate===!0&&e.updateMatrixWorld(),t.parent===null&&t.matrixWorldAutoUpdate===!0&&t.updateMatrixWorld(),Ze.enabled===!0&&Ze.isPresenting===!0&&(w===null||w.isCompositing()===!1)&&(Ze.cameraAutoUpdate===!0&&Ze.updateCamera(t),t=Ze.getCamera()),e.isScene===!0&&e.onBeforeRender(T,e,t,N),x=Be.get(e,C.length),x.init(t),x.state.textureUnits=Me.getTextureUnits(),C.push(x),xe.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),ve.setFromProjectionMatrix(xe,dt,t.reversedDepth),be=this.localClippingEnabled,ye=Ve.init(this.clippingPlanes,be),b=ze.get(e,S.length),b.init(),S.push(b),Ze.enabled===!0&&Ze.isPresenting===!0){let e=T.xr.getDepthSensingMesh();e!==null&&ft(e,t,-1/0,T.sortObjects)}ft(e,t,0,T.sortObjects),b.finish(),T.sortObjects===!0&&b.sort(V,de,t.reversedDepth),Te=Ze.enabled===!1||Ze.isPresenting===!1||Ze.hasDepthSensing()===!1,Te&&Ue.addToRenderList(b,e),this.info.render.frame++,this.info.autoReset===!0&&this.info.reset(),ye===!0&&Ve.beginShadows();let i=x.state.shadowsArray;if(He.render(i,e,t),ye===!0&&Ve.endShadows(),(r&&w.hasRenderPass())===!1){let n=b.opaque,r=b.transmissive;if(x.setupLights(),t.isArrayCamera){let i=t.cameras;if(r.length>0)for(let t=0,a=i.length;t<a;t++){let a=i[t];mt(n,r,e,a)}Te&&Ue.render(e);for(let t=0,n=i.length;t<n;t++){let n=i[t];pt(b,e,n,n.viewport)}}else r.length>0&&mt(n,r,e,t),Te&&Ue.render(e),pt(b,e,t)}N!==null&&M===0&&(Me.updateMultisampleRenderTarget(N),Me.updateRenderTargetMipmap(N)),r&&w.end(T),e.isScene===!0&&e.onAfterRender(T,e,t),Je.resetDefaultState(),P=-1,ee=null,C.pop(),C.length>0?(x=C[C.length-1],Me.setTextureUnits(x.state.textureUnits),ye===!0&&Ve.setGlobalState(T.clippingPlanes,x.state.camera)):x=null,S.pop(),b=S.length>0?S[S.length-1]:null,D!==null&&D.renderEnd()};function ft(e,t,n,r){if(e.visible===!1)return;if(e.layers.test(t.layers)){if(e.isGroup)n=e.renderOrder;else if(e.isLOD)e.autoUpdate===!0&&e.update(t);else if(e.isLightProbeGrid)x.pushLightProbeGrid(e);else if(e.isLight)x.pushLight(e),e.castShadow&&x.pushShadow(e);else if(e.isSprite){if(!e.frustumCulled||ve.intersectsSprite(e)){r&&Ce.setFromMatrixPosition(e.matrixWorld).applyMatrix4(xe);let t=Ie.update(e),i=e.material;i.visible&&b.push(e,t,i,n,Ce.z,null)}}else if((e.isMesh||e.isLine||e.isPoints)&&(!e.frustumCulled||ve.intersectsObject(e))){let t=Ie.update(e),i=e.material;if(r&&(e.boundingSphere===void 0?(t.boundingSphere===null&&t.computeBoundingSphere(),Ce.copy(t.boundingSphere.center)):(e.boundingSphere===null&&e.computeBoundingSphere(),Ce.copy(e.boundingSphere.center)),Ce.applyMatrix4(e.matrixWorld).applyMatrix4(xe)),Array.isArray(i)){let r=t.groups;for(let a=0,o=r.length;a<o;a++){let o=r[a],s=i[o.materialIndex];s&&s.visible&&b.push(e,t,s,n,Ce.z,o)}}else i.visible&&b.push(e,t,i,n,Ce.z,null)}}let i=e.children;for(let e=0,a=i.length;e<a;e++)ft(i[e],t,n,r)}function pt(e,t,n,r){let{opaque:i,transmissive:a,transparent:o}=e;x.setupLightsView(n),ye===!0&&Ve.setGlobalState(T.clippingPlanes,n),r&&U.viewport(te.copy(r)),i.length>0&&gt(i,t,n),a.length>0&&gt(a,t,n),o.length>0&&gt(o,t,n),U.buffers.depth.setTest(!0),U.buffers.depth.setMask(!0),U.buffers.color.setMask(!0),U.setPolygonOffset(!1)}function mt(e,t,n,r){if((n.isScene===!0?n.overrideMaterial:null)!==null)return;if(x.state.transmissionRenderTarget[r.id]===void 0){let e=Oe.has(`EXT_color_buffer_half_float`)||Oe.has(`EXT_color_buffer_float`);x.state.transmissionRenderTarget[r.id]=new Qt(1,1,{generateMipmaps:!0,type:e?ie:I,minFilter:F,samples:Math.max(4,ke.samples),stencilBuffer:i,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:Y.workingColorSpace})}let a=x.state.transmissionRenderTarget[r.id],o=r.viewport||te;a.setSize(o.z*T.transmissionResolutionScale,o.w*T.transmissionResolutionScale);let s=T.getRenderTarget(),c=T.getActiveCubeFace(),l=T.getActiveMipmapLevel();T.setRenderTarget(a),T.getClearColor(se),ce=T.getClearAlpha(),ce<1&&T.setClearColor(16777215,.5),T.clear(),Te&&Ue.render(n);let u=T.toneMapping;T.toneMapping=0;let d=r.viewport;if(r.viewport!==void 0&&(r.viewport=void 0),x.setupLightsView(r),ye===!0&&Ve.setGlobalState(T.clippingPlanes,r),gt(e,n,r),Me.updateMultisampleRenderTarget(a),Me.updateRenderTargetMipmap(a),Oe.has(`WEBGL_multisampled_render_to_texture`)===!1){let e=!1;for(let i=0,a=t.length;i<a;i++){let{object:a,geometry:o,material:s,group:c}=t[i];if(s.side===2&&a.layers.test(r.layers)){let t=s.side;s.side=1,s.needsUpdate=!0,_t(a,n,r,o,s,c),s.side=t,s.needsUpdate=!0,e=!0}}e===!0&&(Me.updateMultisampleRenderTarget(a),Me.updateRenderTargetMipmap(a))}T.setRenderTarget(s,c,l),T.setClearColor(se,ce),d!==void 0&&(r.viewport=d),T.toneMapping=u}function gt(e,t,n){let r=t.isScene===!0?t.overrideMaterial:null;for(let i=0,a=e.length;i<a;i++){let a=e[i],{object:o,geometry:s,group:c}=a,l=a.material;l.allowOverride===!0&&r!==null&&(l=r),o.layers.test(n.layers)&&_t(o,t,n,s,l,c)}}function _t(e,t,n,r,i,a){e.onBeforeRender(T,t,n,r,i,a),e.modelViewMatrix.multiplyMatrices(n.matrixWorldInverse,e.matrixWorld),e.normalMatrix.getNormalMatrix(e.modelViewMatrix),i.onBeforeRender(T,t,n,r,e,a),i.transparent===!0&&i.side===2&&i.forceSinglePass===!1?(i.side=1,i.needsUpdate=!0,T.renderBufferDirect(n,t,r,i,e,a),i.side=0,i.needsUpdate=!0,T.renderBufferDirect(n,t,r,i,e,a),i.side=2):T.renderBufferDirect(n,t,r,i,e,a),e.onAfterRender(T,t,n,r,i,a)}function yt(e,t,n){t.isScene!==!0&&(t=we);let r=je.get(e),i=x.state.lights,a=x.state.shadowsArray,o=i.state.version,s=Le.getParameters(e,i.state,a,t,n,x.state.lightProbeGridArray),c=Le.getProgramCacheKey(s),l=r.programs;r.environment=e.isMeshStandardMaterial||e.isMeshLambertMaterial||e.isMeshPhongMaterial?t.environment:null,r.fog=t.fog;let u=e.isMeshStandardMaterial||e.isMeshLambertMaterial&&!e.envMap||e.isMeshPhongMaterial&&!e.envMap;r.envMap=Ne.get(e.envMap||r.environment,u),r.envMapRotation=r.environment!==null&&e.envMap===null?t.environmentRotation:e.envMapRotation,l===void 0&&(e.addEventListener(`dispose`,tt),l=new Map,r.programs=l);let d=l.get(c);if(d!==void 0){if(r.currentProgram===d&&r.lightsStateVersion===o)return St(e,s),d}else s.uniforms=Le.getUniforms(e),D!==null&&e.isNodeMaterial&&D.build(e,n,s),e.onBeforeCompile(s,T),d=Le.acquireProgram(s,c),l.set(c,d),r.uniforms=s.uniforms;let f=r.uniforms;return(!e.isShaderMaterial&&!e.isRawShaderMaterial||e.clipping===!0)&&(f.clippingPlanes=Ve.uniform),St(e,s),r.needsLights=Et(e),r.lightsStateVersion=o,r.needsLights&&(f.ambientLightColor.value=i.state.ambient,f.lightProbe.value=i.state.probe,f.directionalLights.value=i.state.directional,f.directionalLightShadows.value=i.state.directionalShadow,f.spotLights.value=i.state.spot,f.spotLightShadows.value=i.state.spotShadow,f.rectAreaLights.value=i.state.rectArea,f.ltc_1.value=i.state.rectAreaLTC1,f.ltc_2.value=i.state.rectAreaLTC2,f.pointLights.value=i.state.point,f.pointLightShadows.value=i.state.pointShadow,f.hemisphereLights.value=i.state.hemi,f.directionalShadowMatrix.value=i.state.directionalShadowMatrix,f.spotLightMatrix.value=i.state.spotLightMatrix,f.spotLightMap.value=i.state.spotLightMap,f.pointShadowMatrix.value=i.state.pointShadowMatrix),r.lightProbeGrid=x.state.lightProbeGridArray.length>0,r.currentProgram=d,r.uniformsList=null,d}function bt(e){if(e.uniformsList===null){let t=e.currentProgram.getUniforms();e.uniformsList=$s.seqWithValue(t.seq,e.uniforms)}return e.uniformsList}function St(e,t){let n=je.get(e);n.outputColorSpace=t.outputColorSpace,n.batching=t.batching,n.batchingColor=t.batchingColor,n.instancing=t.instancing,n.instancingColor=t.instancingColor,n.instancingMorph=t.instancingMorph,n.skinning=t.skinning,n.morphTargets=t.morphTargets,n.morphNormals=t.morphNormals,n.morphColors=t.morphColors,n.morphTargetsCount=t.morphTargetsCount,n.numClippingPlanes=t.numClippingPlanes,n.numIntersection=t.numClipIntersection,n.vertexAlphas=t.vertexAlphas,n.vertexTangents=t.vertexTangents,n.toneMapping=t.toneMapping}function Ct(e,t){if(e.length===0)return null;if(e.length===1)return e[0].texture===null?null:e[0];y.setFromMatrixPosition(t.matrixWorld);for(let t=0,n=e.length;t<n;t++){let n=e[t];if(n.texture!==null&&n.boundingBox.containsPoint(y))return n}return null}function wt(e,t,n,r,i){t.isScene!==!0&&(t=we),Me.resetTextureUnits();let a=t.fog,o=r.isMeshStandardMaterial||r.isMeshLambertMaterial||r.isMeshPhongMaterial?t.environment:null,s=N===null?T.outputColorSpace:N.isXRRenderTarget===!0?N.texture.colorSpace:Y.workingColorSpace,c=r.isMeshStandardMaterial||r.isMeshLambertMaterial&&!r.envMap||r.isMeshPhongMaterial&&!r.envMap,l=Ne.get(r.envMap||o,c),u=r.vertexColors===!0&&!!n.attributes.color&&n.attributes.color.itemSize===4,d=!!n.attributes.tangent&&(!!r.normalMap||r.anisotropy>0),f=!!n.morphAttributes.position,p=!!n.morphAttributes.normal,m=!!n.morphAttributes.color,h=0;r.toneMapped&&(N===null||N.isXRRenderTarget===!0)&&(h=T.toneMapping);let g=n.morphAttributes.position||n.morphAttributes.normal||n.morphAttributes.color,_=g===void 0?0:g.length,v=je.get(r),y=x.state.lights;if(ye===!0&&(be===!0||e!==ee)){let t=e===ee&&r.id===P;Ve.setState(r,e,t)}let b=!1;r.version===v.__version?v.needsLights&&v.lightsStateVersion!==y.state.version?b=!0:v.outputColorSpace===s?i.isBatchedMesh&&v.batching===!1||!i.isBatchedMesh&&v.batching===!0||i.isBatchedMesh&&v.batchingColor===!0&&i.colorTexture===null||i.isBatchedMesh&&v.batchingColor===!1&&i.colorTexture!==null||i.isInstancedMesh&&v.instancing===!1||!i.isInstancedMesh&&v.instancing===!0||i.isSkinnedMesh&&v.skinning===!1||!i.isSkinnedMesh&&v.skinning===!0||i.isInstancedMesh&&v.instancingColor===!0&&i.instanceColor===null||i.isInstancedMesh&&v.instancingColor===!1&&i.instanceColor!==null||i.isInstancedMesh&&v.instancingMorph===!0&&i.morphTexture===null||i.isInstancedMesh&&v.instancingMorph===!1&&i.morphTexture!==null?b=!0:v.envMap===l?r.fog===!0&&v.fog!==a||v.numClippingPlanes!==void 0&&(v.numClippingPlanes!==Ve.numPlanes||v.numIntersection!==Ve.numIntersection)?b=!0:v.vertexAlphas===u&&v.vertexTangents===d&&v.morphTargets===f&&v.morphNormals===p&&v.morphColors===m&&v.toneMapping===h&&v.morphTargetsCount===_?!!v.lightProbeGrid!=x.state.lightProbeGridArray.length>0&&(b=!0):b=!0:b=!0:b=!0:(b=!0,v.__version=r.version);let S=v.currentProgram;b===!0&&(S=yt(r,t,i),D&&r.isNodeMaterial&&D.onUpdateProgram(r,S,v));let C=!1,w=!1,E=!1,O=S.getUniforms(),k=v.uniforms;if(U.useProgram(S.program)&&(C=!0,w=!0,E=!0),r.id!==P&&(P=r.id,w=!0),v.needsLights){let e=Ct(x.state.lightProbeGridArray,i);v.lightProbeGrid!==e&&(v.lightProbeGrid=e,w=!0)}if(C||ee!==e){U.buffers.depth.getReversed()&&e.reversedDepth!==!0&&(e._reversedDepth=!0,e.updateProjectionMatrix()),O.setValue(H,`projectionMatrix`,e.projectionMatrix),O.setValue(H,`viewMatrix`,e.matrixWorldInverse);let t=O.map.cameraPosition;t!==void 0&&t.setValue(H,Se.setFromMatrixPosition(e.matrixWorld)),ke.logarithmicDepthBuffer&&O.setValue(H,`logDepthBufFC`,2/(Math.log(e.far+1)/Math.LN2)),(r.isMeshPhongMaterial||r.isMeshToonMaterial||r.isMeshLambertMaterial||r.isMeshBasicMaterial||r.isMeshStandardMaterial||r.isShaderMaterial)&&O.setValue(H,`isOrthographic`,e.isOrthographicCamera===!0),ee!==e&&(ee=e,w=!0,E=!0)}if(v.needsLights&&(y.state.directionalShadowMap.length>0&&O.setValue(H,`directionalShadowMap`,y.state.directionalShadowMap,Me),y.state.spotShadowMap.length>0&&O.setValue(H,`spotShadowMap`,y.state.spotShadowMap,Me),y.state.pointShadowMap.length>0&&O.setValue(H,`pointShadowMap`,y.state.pointShadowMap,Me)),i.isSkinnedMesh){O.setOptional(H,i,`bindMatrix`),O.setOptional(H,i,`bindMatrixInverse`);let e=i.skeleton;e&&(e.boneTexture===null&&e.computeBoneTexture(),O.setValue(H,`boneTexture`,e.boneTexture,Me))}i.isBatchedMesh&&(O.setOptional(H,i,`batchingTexture`),O.setValue(H,`batchingTexture`,i._matricesTexture,Me),O.setOptional(H,i,`batchingIdTexture`),O.setValue(H,`batchingIdTexture`,i._indirectTexture,Me),O.setOptional(H,i,`batchingColorTexture`),i._colorsTexture!==null&&O.setValue(H,`batchingColorTexture`,i._colorsTexture,Me));let A=n.morphAttributes;if((A.position!==void 0||A.normal!==void 0||A.color!==void 0)&&We.update(i,n,S),(w||v.receiveShadow!==i.receiveShadow)&&(v.receiveShadow=i.receiveShadow,O.setValue(H,`receiveShadow`,i.receiveShadow)),(r.isMeshStandardMaterial||r.isMeshLambertMaterial||r.isMeshPhongMaterial)&&r.envMap===null&&t.environment!==null&&(k.envMapIntensity.value=t.environmentIntensity),k.dfgLUT!==void 0&&(k.dfgLUT.value=bl()),w){if(O.setValue(H,`toneMappingExposure`,T.toneMappingExposure),v.needsLights&&Tt(k,E),a&&r.fog===!0&&Re.refreshFogUniforms(k,a),Re.refreshMaterialUniforms(k,r,ue,B,x.state.transmissionRenderTarget[e.id]),v.needsLights&&v.lightProbeGrid){let e=v.lightProbeGrid;k.probesSH.value=e.texture,k.probesMin.value.copy(e.boundingBox.min),k.probesMax.value.copy(e.boundingBox.max),k.probesResolution.value.copy(e.resolution)}$s.upload(H,bt(v),k,Me)}if(r.isShaderMaterial&&r.uniformsNeedUpdate===!0&&($s.upload(H,bt(v),k,Me),r.uniformsNeedUpdate=!1),r.isSpriteMaterial&&O.setValue(H,`center`,i.center),O.setValue(H,`modelViewMatrix`,i.modelViewMatrix),O.setValue(H,`normalMatrix`,i.normalMatrix),O.setValue(H,`modelMatrix`,i.matrixWorld),r.uniformsGroups!==void 0){let e=r.uniformsGroups;for(let t=0,n=e.length;t<n;t++){let n=e[t];Ye.update(n,S),Ye.bind(n,S)}}return S}function Tt(e,t){e.ambientLightColor.needsUpdate=t,e.lightProbe.needsUpdate=t,e.directionalLights.needsUpdate=t,e.directionalLightShadows.needsUpdate=t,e.pointLights.needsUpdate=t,e.pointLightShadows.needsUpdate=t,e.spotLights.needsUpdate=t,e.spotLightShadows.needsUpdate=t,e.rectAreaLights.needsUpdate=t,e.hemisphereLights.needsUpdate=t}function Et(e){return e.isMeshLambertMaterial||e.isMeshToonMaterial||e.isMeshPhongMaterial||e.isMeshStandardMaterial||e.isShadowMaterial||e.isShaderMaterial&&e.lights===!0}this.getActiveCubeFace=function(){return j},this.getActiveMipmapLevel=function(){return M},this.getRenderTarget=function(){return N},this.setRenderTargetTextures=function(e,t,n){let r=je.get(e);r.__autoAllocateDepthBuffer=e.resolveDepthBuffer===!1,r.__autoAllocateDepthBuffer===!1&&(r.__useRenderToTexture=!1),je.get(e.texture).__webglTexture=t,je.get(e.depthTexture).__webglTexture=r.__autoAllocateDepthBuffer?void 0:n,r.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(e,t){let n=je.get(e);n.__webglFramebuffer=t,n.__useDefaultFramebuffer=t===void 0},this.setRenderTarget=function(e,t=0,n=0){N=e,j=t,M=n;let r=null,i=!1,a=!1;if(e){let o=je.get(e);if(o.__useDefaultFramebuffer!==void 0){U.bindFramebuffer(H.FRAMEBUFFER,o.__webglFramebuffer),te.copy(e.viewport),R.copy(e.scissor),re=e.scissorTest,U.viewport(te),U.scissor(R),U.setScissorTest(re),P=-1;return}else if(o.__webglFramebuffer===void 0)Me.setupRenderTarget(e);else if(o.__hasExternalTextures)Me.rebindTextures(e,je.get(e.texture).__webglTexture,je.get(e.depthTexture).__webglTexture);else if(e.depthBuffer){let t=e.depthTexture;if(o.__boundDepthTexture!==t){if(t!==null&&je.has(t)&&(e.width!==t.image.width||e.height!==t.image.height))throw Error(`THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.`);Me.setupDepthRenderbuffer(e)}}let s=e.texture;(s.isData3DTexture||s.isDataArrayTexture||s.isCompressedArrayTexture)&&(a=!0);let c=je.get(e).__webglFramebuffer;e.isWebGLCubeRenderTarget?(r=Array.isArray(c[t])?c[t][n]:c[t],i=!0):r=e.samples>0&&Me.useMultisampledRTT(e)===!1?je.get(e).__webglMultisampledFramebuffer:Array.isArray(c)?c[n]:c,te.copy(e.viewport),R.copy(e.scissor),re=e.scissorTest}else te.copy(fe).multiplyScalar(ue).floor(),R.copy(me).multiplyScalar(ue).floor(),re=_e;if(n!==0&&(r=O),U.bindFramebuffer(H.FRAMEBUFFER,r)&&U.drawBuffers(e,r),U.viewport(te),U.scissor(R),U.setScissorTest(re),i){let r=je.get(e.texture);H.framebufferTexture2D(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_CUBE_MAP_POSITIVE_X+t,r.__webglTexture,n)}else if(a){let r=t;for(let t=0;t<e.textures.length;t++){let i=je.get(e.textures[t]);H.framebufferTextureLayer(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0+t,i.__webglTexture,n,r)}}else if(e!==null&&n!==0){let t=je.get(e.texture);H.framebufferTexture2D(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,t.__webglTexture,n)}P=-1},this.readRenderTargetPixels=function(e,t,n,r,i,a,o,s=0){if(!(e&&e.isWebGLRenderTarget)){G(`WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.`);return}let c=je.get(e).__webglFramebuffer;if(e.isWebGLCubeRenderTarget&&o!==void 0&&(c=c[o]),c){U.bindFramebuffer(H.FRAMEBUFFER,c);try{let o=e.textures[s],c=o.format,l=o.type;if(e.textures.length>1&&H.readBuffer(H.COLOR_ATTACHMENT0+s),!ke.textureFormatReadable(c)){G(`WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.`);return}if(!ke.textureTypeReadable(l)){G(`WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.`);return}t>=0&&t<=e.width-r&&n>=0&&n<=e.height-i&&H.readPixels(t,n,r,i,qe.convert(c),qe.convert(l),a)}finally{let e=N===null?null:je.get(N).__webglFramebuffer;U.bindFramebuffer(H.FRAMEBUFFER,e)}}},this.readRenderTargetPixelsAsync=async function(e,t,n,r,i,a,o,s=0){if(!(e&&e.isWebGLRenderTarget))throw Error(`THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.`);let c=je.get(e).__webglFramebuffer;if(e.isWebGLCubeRenderTarget&&o!==void 0&&(c=c[o]),c)if(t>=0&&t<=e.width-r&&n>=0&&n<=e.height-i){U.bindFramebuffer(H.FRAMEBUFFER,c);let o=e.textures[s],l=o.format,u=o.type;if(e.textures.length>1&&H.readBuffer(H.COLOR_ATTACHMENT0+s),!ke.textureFormatReadable(l))throw Error(`THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.`);if(!ke.textureTypeReadable(u))throw Error(`THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.`);let d=H.createBuffer();H.bindBuffer(H.PIXEL_PACK_BUFFER,d),H.bufferData(H.PIXEL_PACK_BUFFER,a.byteLength,H.STREAM_READ),H.readPixels(t,n,r,i,qe.convert(l),qe.convert(u),0);let f=N===null?null:je.get(N).__webglFramebuffer;U.bindFramebuffer(H.FRAMEBUFFER,f);let p=H.fenceSync(H.SYNC_GPU_COMMANDS_COMPLETE,0);return H.flush(),await xt(H,p,4),H.bindBuffer(H.PIXEL_PACK_BUFFER,d),H.getBufferSubData(H.PIXEL_PACK_BUFFER,0,a),H.deleteBuffer(d),H.deleteSync(p),a}else throw Error(`THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.`)},this.copyFramebufferToTexture=function(e,t=null,n=0){let r=2**-n,i=Math.floor(e.image.width*r),a=Math.floor(e.image.height*r),o=t===null?0:t.x,s=t===null?0:t.y;Me.setTexture2D(e,0),H.copyTexSubImage2D(H.TEXTURE_2D,n,0,0,o,s,i,a),U.unbindTexture()},this.copyTextureToTexture=function(e,t,n=null,r=null,i=0,a=0){let o,s,c,l,u,d,f,p,m,h=e.isCompressedTexture?e.mipmaps[a]:e.image;if(n!==null)o=n.max.x-n.min.x,s=n.max.y-n.min.y,c=n.isBox3?n.max.z-n.min.z:1,l=n.min.x,u=n.min.y,d=n.isBox3?n.min.z:0;else{let t=2**-i;o=Math.floor(h.width*t),s=Math.floor(h.height*t),c=e.isDataArrayTexture?h.depth:e.isData3DTexture?Math.floor(h.depth*t):1,l=0,u=0,d=0}r===null?(f=0,p=0,m=0):(f=r.x,p=r.y,m=r.z);let g=qe.convert(t.format),_=qe.convert(t.type),v;t.isData3DTexture?(Me.setTexture3D(t,0),v=H.TEXTURE_3D):t.isDataArrayTexture||t.isCompressedArrayTexture?(Me.setTexture2DArray(t,0),v=H.TEXTURE_2D_ARRAY):(Me.setTexture2D(t,0),v=H.TEXTURE_2D),U.activeTexture(H.TEXTURE0),U.pixelStorei(H.UNPACK_FLIP_Y_WEBGL,t.flipY),U.pixelStorei(H.UNPACK_PREMULTIPLY_ALPHA_WEBGL,t.premultiplyAlpha),U.pixelStorei(H.UNPACK_ALIGNMENT,t.unpackAlignment);let y=U.getParameter(H.UNPACK_ROW_LENGTH),b=U.getParameter(H.UNPACK_IMAGE_HEIGHT),x=U.getParameter(H.UNPACK_SKIP_PIXELS),S=U.getParameter(H.UNPACK_SKIP_ROWS),C=U.getParameter(H.UNPACK_SKIP_IMAGES);U.pixelStorei(H.UNPACK_ROW_LENGTH,h.width),U.pixelStorei(H.UNPACK_IMAGE_HEIGHT,h.height),U.pixelStorei(H.UNPACK_SKIP_PIXELS,l),U.pixelStorei(H.UNPACK_SKIP_ROWS,u),U.pixelStorei(H.UNPACK_SKIP_IMAGES,d);let w=e.isDataArrayTexture||e.isData3DTexture,T=t.isDataArrayTexture||t.isData3DTexture;if(e.isDepthTexture){let n=je.get(e),r=je.get(t),h=je.get(n.__renderTarget),g=je.get(r.__renderTarget);U.bindFramebuffer(H.READ_FRAMEBUFFER,h.__webglFramebuffer),U.bindFramebuffer(H.DRAW_FRAMEBUFFER,g.__webglFramebuffer);for(let n=0;n<c;n++)w&&(H.framebufferTextureLayer(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,je.get(e).__webglTexture,i,d+n),H.framebufferTextureLayer(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,je.get(t).__webglTexture,a,m+n)),H.blitFramebuffer(l,u,o,s,f,p,o,s,H.DEPTH_BUFFER_BIT,H.NEAREST);U.bindFramebuffer(H.READ_FRAMEBUFFER,null),U.bindFramebuffer(H.DRAW_FRAMEBUFFER,null)}else if(i!==0||e.isRenderTargetTexture||je.has(e)){let n=je.get(e),r=je.get(t);U.bindFramebuffer(H.READ_FRAMEBUFFER,k),U.bindFramebuffer(H.DRAW_FRAMEBUFFER,A);for(let e=0;e<c;e++)w?H.framebufferTextureLayer(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,n.__webglTexture,i,d+e):H.framebufferTexture2D(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,n.__webglTexture,i),T?H.framebufferTextureLayer(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,r.__webglTexture,a,m+e):H.framebufferTexture2D(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,r.__webglTexture,a),i===0?T?H.copyTexSubImage3D(v,a,f,p,m+e,l,u,o,s):H.copyTexSubImage2D(v,a,f,p,l,u,o,s):H.blitFramebuffer(l,u,o,s,f,p,o,s,H.COLOR_BUFFER_BIT,H.NEAREST);U.bindFramebuffer(H.READ_FRAMEBUFFER,null),U.bindFramebuffer(H.DRAW_FRAMEBUFFER,null)}else T?e.isDataTexture||e.isData3DTexture?H.texSubImage3D(v,a,f,p,m,o,s,c,g,_,h.data):t.isCompressedArrayTexture?H.compressedTexSubImage3D(v,a,f,p,m,o,s,c,g,h.data):H.texSubImage3D(v,a,f,p,m,o,s,c,g,_,h):e.isDataTexture?H.texSubImage2D(H.TEXTURE_2D,a,f,p,o,s,g,_,h.data):e.isCompressedTexture?H.compressedTexSubImage2D(H.TEXTURE_2D,a,f,p,h.width,h.height,g,h.data):H.texSubImage2D(H.TEXTURE_2D,a,f,p,o,s,g,_,h);U.pixelStorei(H.UNPACK_ROW_LENGTH,y),U.pixelStorei(H.UNPACK_IMAGE_HEIGHT,b),U.pixelStorei(H.UNPACK_SKIP_PIXELS,x),U.pixelStorei(H.UNPACK_SKIP_ROWS,S),U.pixelStorei(H.UNPACK_SKIP_IMAGES,C),a===0&&t.generateMipmaps&&H.generateMipmap(v),U.unbindTexture()},this.initRenderTarget=function(e){je.get(e).__webglFramebuffer===void 0&&Me.setupRenderTarget(e)},this.initTexture=function(e){e.isCubeTexture?Me.setTextureCube(e,0):e.isData3DTexture?Me.setTexture3D(e,0):e.isDataArrayTexture||e.isCompressedArrayTexture?Me.setTexture2DArray(e,0):Me.setTexture2D(e,0),U.unbindTexture()},this.resetState=function(){j=0,M=0,N=null,U.reset(),Je.reset()},typeof __THREE_DEVTOOLS__<`u`&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent(`observe`,{detail:this}))}get coordinateSystem(){return dt}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let t=this.getContext();t.drawingBufferColorSpace=Y._getDrawingBufferColorSpace(e),t.unpackColorSpace=Y._getUnpackColorSpace()}},Sl=t((e=>{var t=n(),r=i();function a(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}var o=Symbol.for(`react.transitional.element`),s=Symbol.for(`react.portal`),c=Symbol.for(`react.fragment`),l=Symbol.for(`react.strict_mode`),u=Symbol.for(`react.profiler`),d=Symbol.for(`react.consumer`),f=Symbol.for(`react.context`),p=Symbol.for(`react.forward_ref`),m=Symbol.for(`react.suspense`),h=Symbol.for(`react.suspense_list`),g=Symbol.for(`react.memo`),_=Symbol.for(`react.lazy`),v=Symbol.for(`react.scope`),y=Symbol.for(`react.activity`),b=Symbol.for(`react.legacy_hidden`),x=Symbol.for(`react.memo_cache_sentinel`),S=Symbol.for(`react.view_transition`),C=Symbol.iterator;function w(e){return typeof e!=`object`||!e?null:(e=C&&e[C]||e[`@@iterator`],typeof e==`function`?e:null)}var T=Array.isArray;function E(e,t){var n=e.length&3,r=e.length-n,i=t;for(t=0;t<r;){var a=e.charCodeAt(t)&255|(e.charCodeAt(++t)&255)<<8|(e.charCodeAt(++t)&255)<<16|(e.charCodeAt(++t)&255)<<24;++t,a=3432918353*(a&65535)+((3432918353*(a>>>16)&65535)<<16)&4294967295,a=a<<15|a>>>17,a=461845907*(a&65535)+((461845907*(a>>>16)&65535)<<16)&4294967295,i^=a,i=i<<13|i>>>19,i=5*(i&65535)+((5*(i>>>16)&65535)<<16)&4294967295,i=(i&65535)+27492+(((i>>>16)+58964&65535)<<16)}switch(a=0,n){case 3:a^=(e.charCodeAt(t+2)&255)<<16;case 2:a^=(e.charCodeAt(t+1)&255)<<8;case 1:a^=e.charCodeAt(t)&255,a=3432918353*(a&65535)+((3432918353*(a>>>16)&65535)<<16)&4294967295,a=a<<15|a>>>17,i^=461845907*(a&65535)+((461845907*(a>>>16)&65535)<<16)&4294967295}return i^=e.length,i^=i>>>16,i=2246822507*(i&65535)+((2246822507*(i>>>16)&65535)<<16)&4294967295,i^=i>>>13,i=3266489909*(i&65535)+((3266489909*(i>>>16)&65535)<<16)&4294967295,(i^i>>>16)>>>0}var D=Object.assign,O=Object.prototype.hasOwnProperty,k=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),A={},j={};function M(e){return O.call(j,e)?!0:O.call(A,e)?!1:k.test(e)?j[e]=!0:(A[e]=!0,!1)}var N=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `)),P=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),F=/["'&<>]/;function I(e){if(typeof e==`boolean`||typeof e==`number`||typeof e==`bigint`)return``+e;e=``+e;var t=F.exec(e);if(t){var n=``,r,i=0;for(r=t.index;r<e.length;r++){switch(e.charCodeAt(r)){case 34:t=`&quot;`;break;case 38:t=`&amp;`;break;case 39:t=`&#x27;`;break;case 60:t=`&lt;`;break;case 62:t=`&gt;`;break;default:continue}i!==r&&(n+=e.slice(i,r)),i=r+1,n+=t}e=i===r?n:n+e.slice(i,r)}return e}var ee=/([A-Z])/g,te=/^ms-/,L=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function R(e){return L.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}var ne=t.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,re=r.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,ie={pending:!1,data:null,method:null,action:null},z=re.d;re.d={f:z.f,r:z.r,D:ct,C:lt,L:ut,m:dt,X:pt,S:ft,M:mt};var ae=[],oe=null,se=/(<\/|<)(s)(cript)/gi;function ce(e,t,n,r){return``+t+(n===`s`?`\\u0073`:`\\u0053`)+r}function le(e,t,n,r,i){return{idPrefix:e===void 0?``:e,nextFormID:0,streamingFormat:0,bootstrapScriptContent:n,bootstrapScripts:r,bootstrapModules:i,instructions:0,hasBody:!1,hasHtml:!1,unknownResources:{},dnsResources:{},connectResources:{default:{},anonymous:{},credentials:{}},imageResources:{},styleResources:{},scriptResources:{},moduleUnknownResources:{},moduleScriptResources:{}}}function B(e,t,n,r){return{insertionMode:e,selectedValue:t,tagScope:n,viewTransition:r}}function ue(e,t,n){var r=e.tagScope&-25;switch(t){case`noscript`:return B(2,null,r|1,null);case`select`:return B(2,n.value==null?n.defaultValue:n.value,r,null);case`svg`:return B(4,null,r,null);case`picture`:return B(2,null,r|2,null);case`math`:return B(5,null,r,null);case`foreignObject`:return B(2,null,r,null);case`table`:return B(6,null,r,null);case`thead`:case`tbody`:case`tfoot`:return B(7,null,r,null);case`colgroup`:return B(9,null,r,null);case`tr`:return B(8,null,r,null);case`head`:if(2>e.insertionMode)return B(3,null,r,null);break;case`html`:if(e.insertionMode===0)return B(1,null,r,null)}return 6<=e.insertionMode||2>e.insertionMode?B(2,null,r,null):e.tagScope===r?e:B(e.insertionMode,e.selectedValue,r,null)}function V(e){return e===null?null:{update:e.update,enter:`none`,exit:`none`,share:e.update,name:e.autoName,autoName:e.autoName,nameIdx:0}}function de(e,t){return t.tagScope&32&&(e.instructions|=128),B(t.insertionMode,t.selectedValue,t.tagScope|12,V(t.viewTransition))}function fe(e,t){e=V(t.viewTransition);var n=t.tagScope|16;return e!==null&&e.share!==`none`&&(n|=64),B(t.insertionMode,t.selectedValue,n,e)}var pe=new Map;function me(e,t){if(typeof t!=`object`)throw Error(a(62));var n=!0,r;for(r in t)if(O.call(t,r)){var i=t[r];if(i!=null&&typeof i!=`boolean`&&i!==``){if(r.indexOf(`--`)===0){var o=I(r);i=I((``+i).trim())}else o=pe.get(r),o===void 0&&(o=I(r.replace(ee,`-$1`).toLowerCase().replace(te,`-ms-`)),pe.set(r,o)),i=typeof i==`number`?i===0||N.has(r)?``+i:i+`px`:I((``+i).trim());n?(n=!1,e.push(` style="`,o,`:`,i)):e.push(`;`,o,`:`,i)}}n||e.push(`"`)}function he(e,t,n){n&&typeof n!=`function`&&typeof n!=`symbol`&&e.push(` `,t,`=""`)}function ge(e,t,n){typeof n!=`function`&&typeof n!=`symbol`&&typeof n!=`boolean`&&e.push(` `,t,`="`,I(n),`"`)}var _e=I(`javascript:throw new Error('React form unexpectedly submitted.')`);function ve(e,t){this.push(`<input type="hidden"`),ye(e),ge(this,`name`,t),ge(this,`value`,e),this.push(`/>`)}function ye(e){if(typeof e!=`string`)throw Error(a(480))}function be(e,t){if(typeof t.$$FORM_ACTION==`function`){var n=e.nextFormID++;e=e.idPrefix+n;try{var r=t.$$FORM_ACTION(e);return r&&r.data?.forEach(ye),r}catch(e){if(typeof e==`object`&&e&&typeof e.then==`function`)throw e}}return null}function xe(e,t,n,r,i,a,o,s){var c=null;if(typeof r==`function`){var l=be(t,r);l===null?(e.push(` `,`formAction`,`="`,_e,`"`),o=a=i=r=s=null,Te(t,n)):(s=l.name,r=l.action||``,i=l.encType,a=l.method,o=l.target,c=l.data)}return s!=null&&Se(e,`name`,s),r!=null&&Se(e,`formAction`,r),i!=null&&Se(e,`formEncType`,i),a!=null&&Se(e,`formMethod`,a),o!=null&&Se(e,`formTarget`,o),c}function Se(e,t,n){switch(t){case`className`:ge(e,`class`,n);break;case`tabIndex`:ge(e,`tabindex`,n);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:ge(e,t,n);break;case`style`:me(e,n);break;case`src`:case`href`:if(n===``)break;case`action`:case`formAction`:if(n==null||typeof n==`function`||typeof n==`symbol`||typeof n==`boolean`)break;n=R(``+n),e.push(` `,t,`="`,I(n),`"`);break;case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`ref`:break;case`autoFocus`:case`multiple`:case`muted`:he(e,t.toLowerCase(),n);break;case`xlinkHref`:if(typeof n==`function`||typeof n==`symbol`||typeof n==`boolean`)break;n=R(``+n),e.push(` `,`xlink:href`,`="`,I(n),`"`);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:typeof n!=`function`&&typeof n!=`symbol`&&e.push(` `,t,`="`,I(n),`"`);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:n&&typeof n!=`function`&&typeof n!=`symbol`&&e.push(` `,t,`=""`);break;case`capture`:case`download`:!0===n?e.push(` `,t,`=""`):!1!==n&&typeof n!=`function`&&typeof n!=`symbol`&&e.push(` `,t,`="`,I(n),`"`);break;case`cols`:case`rows`:case`size`:case`span`:typeof n!=`function`&&typeof n!=`symbol`&&!isNaN(n)&&1<=n&&e.push(` `,t,`="`,I(n),`"`);break;case`rowSpan`:case`start`:typeof n==`function`||typeof n==`symbol`||isNaN(n)||e.push(` `,t,`="`,I(n),`"`);break;case`xlinkActuate`:ge(e,`xlink:actuate`,n);break;case`xlinkArcrole`:ge(e,`xlink:arcrole`,n);break;case`xlinkRole`:ge(e,`xlink:role`,n);break;case`xlinkShow`:ge(e,`xlink:show`,n);break;case`xlinkTitle`:ge(e,`xlink:title`,n);break;case`xlinkType`:ge(e,`xlink:type`,n);break;case`xmlBase`:ge(e,`xml:base`,n);break;case`xmlLang`:ge(e,`xml:lang`,n);break;case`xmlSpace`:ge(e,`xml:space`,n);break;default:if((!(2<t.length)||t[0]!==`o`&&t[0]!==`O`||t[1]!==`n`&&t[1]!==`N`)&&(t=P.get(t)||t,M(t))){switch(typeof n){case`function`:case`symbol`:return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`)return}e.push(` `,t,`="`,I(n),`"`)}}}function Ce(e,t,n){if(t!=null){if(n!=null)throw Error(a(60));if(typeof t!=`object`||!(`__html`in t))throw Error(a(61));t=t.__html,t!=null&&e.push(``+t)}}function we(e){var n=``;return t.Children.forEach(e,function(e){e!=null&&(n+=e)}),n}function Te(e,t){if(!(e.instructions&16)){e.instructions|=16;var n=t.preamble,r=t.bootstrapChunks;(n.htmlChunks||n.headChunks)&&r.length===0?(r.push(t.startInlineScript),it(r,e),r.push(`>`,`addEventListener("submit",function(a){if(!a.defaultPrevented){var c=a.target,d=a.submitter,e=c.action,b=d;if(d){var f=d.getAttribute("formAction");null!=f&&(e=f,b=null)}"javascript:throw new Error('React form unexpectedly submitted.')"===e&&(a.preventDefault(),b?(a=document.createElement("input"),a.name=b.name,a.value=b.value,b.parentNode.insertBefore(a,b),b=new FormData(c),a.parentNode.removeChild(a)):b=new FormData(c),a=c.ownerDocument||c,(a.$$reactFormReplay=a.$$reactFormReplay||[]).push(c,d,b))}});`,`<\/script>`)):r.unshift(t.startInlineScript,`>`,`addEventListener("submit",function(a){if(!a.defaultPrevented){var c=a.target,d=a.submitter,e=c.action,b=d;if(d){var f=d.getAttribute("formAction");null!=f&&(e=f,b=null)}"javascript:throw new Error('React form unexpectedly submitted.')"===e&&(a.preventDefault(),b?(a=document.createElement("input"),a.name=b.name,a.value=b.value,b.parentNode.insertBefore(a,b),b=new FormData(c),a.parentNode.removeChild(a)):b=new FormData(c),a=c.ownerDocument||c,(a.$$reactFormReplay=a.$$reactFormReplay||[]).push(c,d,b))}});`,`<\/script>`)}}function Ee(e,t){for(var n in e.push(Pe(`link`)),t)if(O.call(t,n)){var r=t[n];if(r!=null)switch(n){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,`link`));default:Se(e,n,r)}}return e.push(`/>`),null}var H=/(<\/|<)(s)(tyle)/gi;function De(e,t,n,r){return``+t+(n===`s`?`\\73 `:`\\53 `)+r}function Oe(e,t,n){for(var r in e.push(Pe(n)),t)if(O.call(t,r)){var i=t[r];if(i!=null)switch(r){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,n));default:Se(e,r,i)}}return e.push(`/>`),null}function ke(e,t){e.push(Pe(`title`));var n=null,r=null,i;for(i in t)if(O.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:Se(e,i,a)}}return e.push(`>`),t=Array.isArray(n)?2>n.length?n[0]:null:n,typeof t!=`function`&&typeof t!=`symbol`&&t!=null&&e.push(I(``+t)),Ce(e,r,n),e.push(Le(`title`)),null}function U(e,t){e.push(Pe(`script`));var n=null,r=null,i;for(i in t)if(O.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:Se(e,i,a)}}return e.push(`>`),Ce(e,r,n),typeof n==`string`&&e.push((``+n).replace(se,ce)),e.push(Le(`script`)),null}function Ae(e,t,n){e.push(Pe(n));var r=n=null,i;for(i in t)if(O.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:Se(e,i,a)}}return e.push(`>`),Ce(e,r,n),n}function je(e,t,n){e.push(Pe(n));var r=n=null,i;for(i in t)if(O.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:Se(e,i,a)}}return e.push(`>`),Ce(e,r,n),typeof n==`string`?(e.push(I(n)),null):n}var Me=/^[a-zA-Z][a-zA-Z:_\.\-\d]*$/,Ne=new Map;function Pe(e){var t=Ne.get(e);if(t===void 0){if(!Me.test(e))throw Error(a(65,e));t=`<`+e,Ne.set(e,t)}return t}function Fe(e,t,n,r,i,o,s,c,l){switch(t){case`div`:case`span`:case`svg`:case`path`:break;case`a`:e.push(Pe(`a`));var u=null,d=null,f;for(f in n)if(O.call(n,f)){var p=n[f];if(p!=null)switch(f){case`children`:u=p;break;case`dangerouslySetInnerHTML`:d=p;break;case`href`:p===``?ge(e,`href`,``):Se(e,f,p);break;default:Se(e,f,p)}}if(e.push(`>`),Ce(e,d,u),typeof u==`string`){e.push(I(u));var m=null}else m=u;return m;case`g`:case`p`:case`li`:break;case`select`:e.push(Pe(`select`));var h=null,g=null,_;for(_ in n)if(O.call(n,_)){var v=n[_];if(v!=null)switch(_){case`children`:h=v;break;case`dangerouslySetInnerHTML`:g=v;break;case`defaultValue`:case`value`:break;default:Se(e,_,v)}}return e.push(`>`),Ce(e,g,h),h;case`option`:var y=c.selectedValue;e.push(Pe(`option`));var b=null,x=null,S=null,C=null,w;for(w in n)if(O.call(n,w)){var E=n[w];if(E!=null)switch(w){case`children`:b=E;break;case`selected`:S=E;break;case`dangerouslySetInnerHTML`:C=E;break;case`value`:x=E;default:Se(e,w,E)}}if(y!=null){var k=x===null?we(b):``+x;if(T(y)){for(var A=0;A<y.length;A++)if(``+y[A]===k){e.push(` selected=""`);break}}else ``+y===k&&e.push(` selected=""`)}else S&&e.push(` selected=""`);return e.push(`>`),Ce(e,C,b),b;case`textarea`:e.push(Pe(`textarea`));var j=null,N=null,P=null,F;for(F in n)if(O.call(n,F)){var ee=n[F];if(ee!=null)switch(F){case`children`:P=ee;break;case`value`:j=ee;break;case`defaultValue`:N=ee;break;case`dangerouslySetInnerHTML`:throw Error(a(91));default:Se(e,F,ee)}}if(j===null&&N!==null&&(j=N),e.push(`>`),P!=null){if(j!=null)throw Error(a(92));if(T(P)){if(1<P.length)throw Error(a(93));j=``+P[0]}j=``+P}return typeof j==`string`&&j[0]===`
`&&e.push(`
`),j!==null&&e.push(I(``+j)),null;case`input`:e.push(Pe(`input`));var te=null,L=null,ne=null,re=null,ie=null,z=null,oe=null,se=null,ce=null,le;for(le in n)if(O.call(n,le)){var B=n[le];if(B!=null)switch(le){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,`input`));case`name`:te=B;break;case`formAction`:L=B;break;case`formEncType`:ne=B;break;case`formMethod`:re=B;break;case`formTarget`:ie=B;break;case`defaultChecked`:ce=B;break;case`defaultValue`:oe=B;break;case`checked`:se=B;break;case`value`:z=B;break;default:Se(e,le,B)}}var ue=xe(e,r,i,L,ne,re,ie,te);return se===null?ce!==null&&he(e,`checked`,ce):he(e,`checked`,se),z===null?oe!==null&&Se(e,`value`,oe):Se(e,`value`,z),e.push(`/>`),ue?.forEach(ve,e),null;case`button`:e.push(Pe(`button`));var V=null,de=null,fe=null,pe=null,ye=null,Me=null,Ne=null,Fe;for(Fe in n)if(O.call(n,Fe)){var Ie=n[Fe];if(Ie!=null)switch(Fe){case`children`:V=Ie;break;case`dangerouslySetInnerHTML`:de=Ie;break;case`name`:fe=Ie;break;case`formAction`:pe=Ie;break;case`formEncType`:ye=Ie;break;case`formMethod`:Me=Ie;break;case`formTarget`:Ne=Ie;break;default:Se(e,Fe,Ie)}}var Re=xe(e,r,i,pe,ye,Me,Ne,fe);if(e.push(`>`),Re?.forEach(ve,e),Ce(e,de,V),typeof V==`string`){e.push(I(V));var ze=null}else ze=V;return ze;case`form`:e.push(Pe(`form`));var Be=null,Ve=null,He=null,Ue=null,We=null,Ge=null,Ke;for(Ke in n)if(O.call(n,Ke)){var qe=n[Ke];if(qe!=null)switch(Ke){case`children`:Be=qe;break;case`dangerouslySetInnerHTML`:Ve=qe;break;case`action`:He=qe;break;case`encType`:Ue=qe;break;case`method`:We=qe;break;case`target`:Ge=qe;break;default:Se(e,Ke,qe)}}var Je=null,Ye=null;if(typeof He==`function`){var Xe=be(r,He);Xe===null?(e.push(` `,`action`,`="`,_e,`"`),Ge=We=Ue=He=null,Te(r,i)):(He=Xe.action||``,Ue=Xe.encType,We=Xe.method,Ge=Xe.target,Je=Xe.data,Ye=Xe.name)}if(He!=null&&Se(e,`action`,He),Ue!=null&&Se(e,`encType`,Ue),We!=null&&Se(e,`method`,We),Ge!=null&&Se(e,`target`,Ge),e.push(`>`),Ye!==null&&(e.push(`<input type="hidden"`),ge(e,`name`,Ye),e.push(`/>`),Je?.forEach(ve,e)),Ce(e,Ve,Be),typeof Be==`string`){e.push(I(Be));var Ze=null}else Ze=Be;return Ze;case`menuitem`:for(var Qe in e.push(Pe(`menuitem`)),n)if(O.call(n,Qe)){var $e=n[Qe];if($e!=null)switch(Qe){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(400));default:Se(e,Qe,$e)}}return e.push(`>`),null;case`object`:e.push(Pe(`object`));var et=null,tt=null,nt;for(nt in n)if(O.call(n,nt)){var rt=n[nt];if(rt!=null)switch(nt){case`children`:et=rt;break;case`dangerouslySetInnerHTML`:tt=rt;break;case`data`:var it=R(``+rt);if(it===``)break;e.push(` `,`data`,`="`,I(it),`"`);break;default:Se(e,nt,rt)}}if(e.push(`>`),Ce(e,tt,et),typeof et==`string`){e.push(I(et));var at=null}else at=et;return at;case`title`:var ot=c.tagScope&1,st=c.tagScope&4;if(c.insertionMode===4||ot||n.itemProp!=null)var ct=ke(e,n);else st?ct=null:(ke(i.hoistableChunks,n),ct=void 0);return ct;case`link`:var lt=c.tagScope&1,ut=c.tagScope&4,dt=n.rel,ft=n.href,pt=n.precedence;if(c.insertionMode===4||lt||n.itemProp!=null||typeof dt!=`string`||typeof ft!=`string`||ft===``){Ee(e,n);var mt=null}else if(n.rel===`stylesheet`)if(typeof pt!=`string`||n.disabled!=null||n.onLoad||n.onError)mt=Ee(e,n);else{var _t=i.styles.get(pt),vt=r.styleResources.hasOwnProperty(ft)?r.styleResources[ft]:void 0;if(vt!==null){r.styleResources[ft]=null,_t||(_t={precedence:I(pt),rules:[],hrefs:[],sheets:new Map},i.styles.set(pt,_t));var yt={state:0,props:D({},n,{"data-precedence":n.precedence,precedence:null})};if(vt){vt.length===2&&ht(yt.props,vt);var W=i.preloads.stylesheets.get(ft);W&&0<W.length?W.length=0:yt.state=1}_t.sheets.set(ft,yt),s&&s.stylesheets.add(yt)}else if(_t){var G=_t.sheets.get(ft);G&&s&&s.stylesheets.add(G)}l&&e.push(`<!-- -->`),mt=null}else n.onLoad||n.onError?mt=Ee(e,n):(l&&e.push(`<!-- -->`),mt=ut?null:Ee(i.hoistableChunks,n));return mt;case`script`:var bt=c.tagScope&1,xt=n.async;if(typeof n.src!=`string`||!n.src||!xt||typeof xt==`function`||typeof xt==`symbol`||n.onLoad||n.onError||c.insertionMode===4||bt||n.itemProp!=null)var St=U(e,n);else{var Ct=n.src;if(n.type===`module`)var wt=r.moduleScriptResources,Tt=i.preloads.moduleScripts;else wt=r.scriptResources,Tt=i.preloads.scripts;var Et=wt.hasOwnProperty(Ct)?wt[Ct]:void 0;if(Et!==null){wt[Ct]=null;var Dt=n;if(Et){Et.length===2&&(Dt=D({},n),ht(Dt,Et));var K=Tt.get(Ct);K&&(K.length=0)}var Ot=[];i.scripts.add(Ot),U(Ot,Dt)}l&&e.push(`<!-- -->`),St=null}return St;case`style`:var kt=c.tagScope&1,At=n.precedence,jt=n.href,Mt=n.nonce;if(c.insertionMode===4||kt||n.itemProp!=null||typeof At!=`string`||typeof jt!=`string`||jt===``){e.push(Pe(`style`));var Nt=null,q=null,Pt;for(Pt in n)if(O.call(n,Pt)){var Ft=n[Pt];if(Ft!=null)switch(Pt){case`children`:Nt=Ft;break;case`dangerouslySetInnerHTML`:q=Ft;break;default:Se(e,Pt,Ft)}}e.push(`>`);var J=Array.isArray(Nt)?2>Nt.length?Nt[0]:null:Nt;typeof J!=`function`&&typeof J!=`symbol`&&J!=null&&e.push((``+J).replace(H,De)),Ce(e,q,Nt),e.push(Le(`style`));var It=null}else{var Lt=i.styles.get(At);if((r.styleResources.hasOwnProperty(jt)?r.styleResources[jt]:void 0)!==null){r.styleResources[jt]=null,Lt||(Lt={precedence:I(At),rules:[],hrefs:[],sheets:new Map},i.styles.set(At,Lt));var Rt=i.nonce.style;if(!Rt||Rt===Mt){Lt.hrefs.push(I(jt));var zt=Lt.rules,Y=null,Bt=null,Vt;for(Vt in n)if(O.call(n,Vt)){var Ht=n[Vt];if(Ht!=null)switch(Vt){case`children`:Y=Ht;break;case`dangerouslySetInnerHTML`:Bt=Ht}}var Ut=Array.isArray(Y)?2>Y.length?Y[0]:null:Y;typeof Ut!=`function`&&typeof Ut!=`symbol`&&Ut!=null&&zt.push((``+Ut).replace(H,De)),Ce(zt,Bt,Y)}}Lt&&s&&s.styles.add(Lt),l&&e.push(`<!-- -->`),It=void 0}return It;case`meta`:var Wt=c.tagScope&1,Gt=c.tagScope&4;if(c.insertionMode===4||Wt||n.itemProp!=null)var Kt=Oe(e,n,`meta`);else l&&e.push(`<!-- -->`),Kt=Gt?null:typeof n.charSet==`string`?Oe(i.charsetChunks,n,`meta`):n.name===`viewport`?Oe(i.viewportChunks,n,`meta`):Oe(i.hoistableChunks,n,`meta`);return Kt;case`listing`:case`pre`:e.push(Pe(t));var qt=null,Jt=null,Yt;for(Yt in n)if(O.call(n,Yt)){var Xt=n[Yt];if(Xt!=null)switch(Yt){case`children`:qt=Xt;break;case`dangerouslySetInnerHTML`:Jt=Xt;break;default:Se(e,Yt,Xt)}}if(e.push(`>`),Jt!=null){if(qt!=null)throw Error(a(60));if(typeof Jt!=`object`||!(`__html`in Jt))throw Error(a(61));var Zt=Jt.__html;Zt!=null&&(typeof Zt==`string`&&0<Zt.length&&Zt[0]===`
`?e.push(`
`,Zt):e.push(``+Zt))}return typeof qt==`string`&&qt[0]===`
`&&e.push(`
`),qt;case`img`:var Qt=c.tagScope&3,$t=n.src,en=n.srcSet;if(!(n.loading===`lazy`||!$t&&!en||typeof $t!=`string`&&$t!=null||typeof en!=`string`&&en!=null||n.fetchPriority===`low`||Qt)&&(typeof $t!=`string`||$t[4]!==`:`||$t[0]!==`d`&&$t[0]!==`D`||$t[1]!==`a`&&$t[1]!==`A`||$t[2]!==`t`&&$t[2]!==`T`||$t[3]!==`a`&&$t[3]!==`A`)&&(typeof en!=`string`||en[4]!==`:`||en[0]!==`d`&&en[0]!==`D`||en[1]!==`a`&&en[1]!==`A`||en[2]!==`t`&&en[2]!==`T`||en[3]!==`a`&&en[3]!==`A`)){s!==null&&c.tagScope&64&&(s.suspenseyImages=!0);var tn=typeof n.sizes==`string`?n.sizes:void 0,nn=en?en+`
`+(tn||``):$t,rn=i.preloads.images,an=rn.get(nn);if(an)(n.fetchPriority===`high`||10>i.highImagePreloads.size)&&(rn.delete(nn),i.highImagePreloads.add(an));else if(!r.imageResources.hasOwnProperty(nn)){r.imageResources[nn]=ae;var on=n.crossOrigin,sn=typeof on==`string`?on===`use-credentials`?on:``:void 0,cn=i.headers,ln;cn&&0<cn.remainingCapacity&&typeof n.srcSet!=`string`&&(n.fetchPriority===`high`||500>cn.highImagePreloads.length)&&(ln=gt($t,`image`,{imageSrcSet:n.srcSet,imageSizes:n.sizes,crossOrigin:sn,integrity:n.integrity,nonce:n.nonce,type:n.type,fetchPriority:n.fetchPriority,referrerPolicy:n.refererPolicy}),0<=(cn.remainingCapacity-=ln.length+2))?(i.resets.image[nn]=ae,cn.highImagePreloads&&(cn.highImagePreloads+=`, `),cn.highImagePreloads+=ln):(an=[],Ee(an,{rel:`preload`,as:`image`,href:en?void 0:$t,imageSrcSet:en,imageSizes:tn,crossOrigin:sn,integrity:n.integrity,type:n.type,fetchPriority:n.fetchPriority,referrerPolicy:n.referrerPolicy}),n.fetchPriority===`high`||10>i.highImagePreloads.size?i.highImagePreloads.add(an):(i.bulkPreloads.add(an),rn.set(nn,an)))}}return Oe(e,n,`img`);case`base`:case`area`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`param`:case`source`:case`track`:case`wbr`:return Oe(e,n,t);case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:break;case`head`:if(2>c.insertionMode){var un=o||i.preamble;if(un.headChunks)throw Error(a(545,"`<head>`"));o!==null&&e.push(`<!--head-->`),un.headChunks=[];var dn=Ae(un.headChunks,n,`head`)}else dn=je(e,n,`head`);return dn;case`body`:if(2>c.insertionMode){var fn=o||i.preamble;if(fn.bodyChunks)throw Error(a(545,"`<body>`"));o!==null&&e.push(`<!--body-->`),fn.bodyChunks=[];var pn=Ae(fn.bodyChunks,n,`body`)}else pn=je(e,n,`body`);return pn;case`html`:if(c.insertionMode===0){var mn=o||i.preamble;if(mn.htmlChunks)throw Error(a(545,"`<html>`"));o!==null&&e.push(`<!--html-->`),mn.htmlChunks=[``];var hn=Ae(mn.htmlChunks,n,`html`)}else hn=je(e,n,`html`);return hn;default:if(t.indexOf(`-`)!==-1){e.push(Pe(t));var gn=null,_n=null,vn;for(vn in n)if(O.call(n,vn)){var yn=n[vn];if(yn!=null){var bn=vn;switch(vn){case`children`:gn=yn;break;case`dangerouslySetInnerHTML`:_n=yn;break;case`style`:me(e,yn);break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`ref`:break;case`className`:bn=`class`;default:if(M(vn)&&typeof yn!=`function`&&typeof yn!=`symbol`&&!1!==yn){if(!0===yn)yn=``;else if(typeof yn==`object`)continue;e.push(` `,bn,`="`,I(yn),`"`)}}}}return e.push(`>`),Ce(e,_n,gn),gn}}return je(e,n,t)}var Ie=new Map;function Le(e){var t=Ie.get(e);return t===void 0&&(t=`</`+e+`>`,Ie.set(e,t)),t}function Re(e,t){e=e.preamble,e.htmlChunks===null&&t.htmlChunks&&(e.htmlChunks=t.htmlChunks),e.headChunks===null&&t.headChunks&&(e.headChunks=t.headChunks),e.bodyChunks===null&&t.bodyChunks&&(e.bodyChunks=t.bodyChunks)}function ze(e,t){t=t.bootstrapChunks;for(var n=0;n<t.length-1;n++)e.push(t[n]);return n<t.length?(n=t[n],t.length=0,e.push(n)):!0}function Be(e,t,n){if(e.push(`<!--$?--><template id="`),n===null)throw Error(a(395));return e.push(t.boundaryPrefix),t=n.toString(16),e.push(t),e.push(`"></template>`)}function Ve(e,t,n,r){switch(n.insertionMode){case 0:case 1:case 3:case 2:return e.push(`<div hidden id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);case 4:return e.push(`<svg aria-hidden="true" style="display:none" id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);case 5:return e.push(`<math aria-hidden="true" style="display:none" id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);case 6:return e.push(`<table hidden id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);case 7:return e.push(`<table hidden><tbody id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);case 8:return e.push(`<table hidden><tr id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);case 9:return e.push(`<table hidden><colgroup id="`),e.push(t.segmentPrefix),t=r.toString(16),e.push(t),e.push(`">`);default:throw Error(a(397))}}function He(e,t){switch(t.insertionMode){case 0:case 1:case 3:case 2:return e.push(`</div>`);case 4:return e.push(`</svg>`);case 5:return e.push(`</math>`);case 6:return e.push(`</table>`);case 7:return e.push(`</tbody></table>`);case 8:return e.push(`</tr></table>`);case 9:return e.push(`</colgroup></table>`);default:throw Error(a(397))}}var Ue=/[<\u2028\u2029]/g;function We(e){return JSON.stringify(e).replace(Ue,function(e){switch(e){case`<`:return`\\u003c`;case`\u2028`:return`\\u2028`;case`\u2029`:return`\\u2029`;default:throw Error(`escapeJSStringsForInstructionScripts encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}})}var Ge=/[&><\u2028\u2029]/g;function Ke(e){return JSON.stringify(e).replace(Ge,function(e){switch(e){case`&`:return`\\u0026`;case`>`:return`\\u003e`;case`<`:return`\\u003c`;case`\u2028`:return`\\u2028`;case`\u2029`:return`\\u2029`;default:throw Error(`escapeJSObjectForInstructionScripts encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}})}var qe=!1,Je=!0;function Ye(e){var t=e.rules,n=e.hrefs,r=0;if(n.length){for(this.push(oe.startInlineStyle),this.push(` media="not all" data-precedence="`),this.push(e.precedence),this.push(`" data-href="`);r<n.length-1;r++)this.push(n[r]),this.push(` `);for(this.push(n[r]),this.push(`">`),r=0;r<t.length;r++)this.push(t[r]);Je=this.push(`</style>`),qe=!0,t.length=0,n.length=0}}function Xe(e){return e.state===2?!1:qe=!0}function Ze(e,t,n){return qe=!1,Je=!0,oe=n,t.styles.forEach(Ye,e),oe=null,t.stylesheets.forEach(Xe),qe&&(n.stylesToHoist=!0),Je}function Qe(e){for(var t=0;t<e.length;t++)this.push(e[t]);e.length=0}var $e=[];function et(e){Ee($e,e.props);for(var t=0;t<$e.length;t++)this.push($e[t]);$e.length=0,e.state=2}function tt(e){var t=0<e.sheets.size;e.sheets.forEach(et,this),e.sheets.clear();var n=e.rules,r=e.hrefs;if(!t||r.length){if(this.push(oe.startInlineStyle),this.push(` data-precedence="`),this.push(e.precedence),e=0,r.length){for(this.push(`" data-href="`);e<r.length-1;e++)this.push(r[e]),this.push(` `);this.push(r[e])}for(this.push(`">`),e=0;e<n.length;e++)this.push(n[e]);this.push(`</style>`),n.length=0,r.length=0}}function nt(e){if(e.state===0){e.state=1;var t=e.props;for(Ee($e,{rel:`preload`,as:`style`,href:e.props.href,crossOrigin:t.crossOrigin,fetchPriority:t.fetchPriority,integrity:t.integrity,media:t.media,hrefLang:t.hrefLang,referrerPolicy:t.referrerPolicy}),e=0;e<$e.length;e++)this.push($e[e]);$e.length=0}}function rt(e){e.sheets.forEach(nt,this),e.sheets.clear()}function it(e,t){!(t.instructions&32)&&(t.instructions|=32,e.push(` id="`,I(`_`+t.idPrefix+`R_`),`"`))}function at(e,t){e.push(`[`);var n=`[`;t.stylesheets.forEach(function(t){if(t.state!==2)if(t.state===3)e.push(n),t=Ke(``+t.props.href),e.push(t),e.push(`]`),n=`,[`;else{e.push(n);var r=t.props[`data-precedence`],i=t.props,o=R(``+t.props.href);for(var s in o=Ke(o),e.push(o),r=``+r,e.push(`,`),r=Ke(r),e.push(r),i)if(O.call(i,s)&&(r=i[s],r!=null))switch(s){case`href`:case`rel`:case`precedence`:case`data-precedence`:break;case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,`link`));default:ot(e,s,r)}e.push(`]`),n=`,[`,t.state=3}}),e.push(`]`)}function ot(e,t,n){var r=t.toLowerCase();switch(typeof n){case`function`:case`symbol`:return}switch(t){case`innerHTML`:case`dangerouslySetInnerHTML`:case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`style`:case`ref`:return;case`className`:r=`class`,t=``+n;break;case`hidden`:if(!1===n)return;t=``;break;case`src`:case`href`:n=R(n),t=``+n;break;default:if(2<t.length&&(t[0]===`o`||t[0]===`O`)&&(t[1]===`n`||t[1]===`N`)||!M(t))return;t=``+n}e.push(`,`),r=Ke(r),e.push(r),e.push(`,`),r=Ke(t),e.push(r)}function st(){return{styles:new Set,stylesheets:new Set,suspenseyImages:!1}}function ct(e){var t=In||null;if(t){var n=t.resumableState,r=t.renderState;if(typeof e==`string`&&e){if(!n.dnsResources.hasOwnProperty(e)){n.dnsResources[e]=null,n=r.headers;var i,a;(a=n&&0<n.remainingCapacity)&&(a=(i=`<`+(``+e).replace(_t,vt)+`>; rel=dns-prefetch`,0<=(n.remainingCapacity-=i.length+2))),a?(r.resets.dns[e]=null,n.preconnects&&(n.preconnects+=`, `),n.preconnects+=i):(i=[],Ee(i,{href:e,rel:`dns-prefetch`}),r.preconnects.add(i))}jr(t)}}else z.D(e)}function lt(e,t){var n=In||null;if(n){var r=n.resumableState,i=n.renderState;if(typeof e==`string`&&e){var a=t===`use-credentials`?`credentials`:typeof t==`string`?`anonymous`:`default`;if(!r.connectResources[a].hasOwnProperty(e)){r.connectResources[a][e]=null,r=i.headers;var o,s;if(s=r&&0<r.remainingCapacity){if(s=`<`+(``+e).replace(_t,vt)+`>; rel=preconnect`,typeof t==`string`){var c=(``+t).replace(yt,W);s+=`; crossorigin="`+c+`"`}s=(o=s,0<=(r.remainingCapacity-=o.length+2))}s?(i.resets.connect[a][e]=null,r.preconnects&&(r.preconnects+=`, `),r.preconnects+=o):(a=[],Ee(a,{rel:`preconnect`,href:e,crossOrigin:t}),i.preconnects.add(a))}jr(n)}}else z.C(e,t)}function ut(e,t,n){var r=In||null;if(r){var i=r.resumableState,a=r.renderState;if(t&&e){switch(t){case`image`:if(n)var o=n.imageSrcSet,s=n.imageSizes,c=n.fetchPriority;var l=o?o+`
`+(s||``):e;if(i.imageResources.hasOwnProperty(l))return;i.imageResources[l]=ae,i=a.headers;var u;i&&0<i.remainingCapacity&&typeof o!=`string`&&c===`high`&&(u=gt(e,t,n),0<=(i.remainingCapacity-=u.length+2))?(a.resets.image[l]=ae,i.highImagePreloads&&(i.highImagePreloads+=`, `),i.highImagePreloads+=u):(i=[],Ee(i,D({rel:`preload`,href:o?void 0:e,as:t},n)),c===`high`?a.highImagePreloads.add(i):(a.bulkPreloads.add(i),a.preloads.images.set(l,i)));break;case`style`:if(i.styleResources.hasOwnProperty(e))return;o=[],Ee(o,D({rel:`preload`,href:e,as:t},n)),i.styleResources[e]=!n||typeof n.crossOrigin!=`string`&&typeof n.integrity!=`string`?ae:[n.crossOrigin,n.integrity],a.preloads.stylesheets.set(e,o),a.bulkPreloads.add(o);break;case`script`:if(i.scriptResources.hasOwnProperty(e))return;o=[],a.preloads.scripts.set(e,o),a.bulkPreloads.add(o),Ee(o,D({rel:`preload`,href:e,as:t},n)),i.scriptResources[e]=!n||typeof n.crossOrigin!=`string`&&typeof n.integrity!=`string`?ae:[n.crossOrigin,n.integrity];break;default:if(i.unknownResources.hasOwnProperty(t)){if(o=i.unknownResources[t],o.hasOwnProperty(e))return}else o={},i.unknownResources[t]=o;if(o[e]=ae,(i=a.headers)&&0<i.remainingCapacity&&t===`font`&&(l=gt(e,t,n),0<=(i.remainingCapacity-=l.length+2)))a.resets.font[e]=ae,i.fontPreloads&&(i.fontPreloads+=`, `),i.fontPreloads+=l;else switch(i=[],e=D({rel:`preload`,href:e,as:t},n),Ee(i,e),t){case`font`:a.fontPreloads.add(i);break;default:a.bulkPreloads.add(i)}}jr(r)}}else z.L(e,t,n)}function dt(e,t){var n=In||null;if(n){var r=n.resumableState,i=n.renderState;if(e){var a=t&&typeof t.as==`string`?t.as:`script`;switch(a){case`script`:if(r.moduleScriptResources.hasOwnProperty(e))return;a=[],r.moduleScriptResources[e]=!t||typeof t.crossOrigin!=`string`&&typeof t.integrity!=`string`?ae:[t.crossOrigin,t.integrity],i.preloads.moduleScripts.set(e,a);break;default:if(r.moduleUnknownResources.hasOwnProperty(a)){var o=r.unknownResources[a];if(o.hasOwnProperty(e))return}else o={},r.moduleUnknownResources[a]=o;a=[],o[e]=ae}Ee(a,D({rel:`modulepreload`,href:e},t)),i.bulkPreloads.add(a),jr(n)}}else z.m(e,t)}function ft(e,t,n){var r=In||null;if(r){var i=r.resumableState,a=r.renderState;if(e){t||=`default`;var o=a.styles.get(t),s=i.styleResources.hasOwnProperty(e)?i.styleResources[e]:void 0;s!==null&&(i.styleResources[e]=null,o||(o={precedence:I(t),rules:[],hrefs:[],sheets:new Map},a.styles.set(t,o)),t={state:0,props:D({rel:`stylesheet`,href:e,"data-precedence":t},n)},s&&(s.length===2&&ht(t.props,s),(a=a.preloads.stylesheets.get(e))&&0<a.length?a.length=0:t.state=1),o.sheets.set(e,t),jr(r))}}else z.S(e,t,n)}function pt(e,t){var n=In||null;if(n){var r=n.resumableState,i=n.renderState;if(e){var a=r.scriptResources.hasOwnProperty(e)?r.scriptResources[e]:void 0;a!==null&&(r.scriptResources[e]=null,t=D({src:e,async:!0},t),a&&(a.length===2&&ht(t,a),e=i.preloads.scripts.get(e))&&(e.length=0),e=[],i.scripts.add(e),U(e,t),jr(n))}}else z.X(e,t)}function mt(e,t){var n=In||null;if(n){var r=n.resumableState,i=n.renderState;if(e){var a=r.moduleScriptResources.hasOwnProperty(e)?r.moduleScriptResources[e]:void 0;a!==null&&(r.moduleScriptResources[e]=null,t=D({src:e,type:`module`,async:!0},t),a&&(a.length===2&&ht(t,a),e=i.preloads.moduleScripts.get(e))&&(e.length=0),e=[],i.scripts.add(e),U(e,t),jr(n))}}else z.M(e,t)}function ht(e,t){e.crossOrigin??=t[0],e.integrity??=t[1]}function gt(e,t,n){for(var r in e=(``+e).replace(_t,vt),t=(``+t).replace(yt,W),t=`<`+e+`>; rel=preload; as="`+t+`"`,n)O.call(n,r)&&(e=n[r],typeof e==`string`&&(t+=`; `+r.toLowerCase()+`="`+(``+e).replace(yt,W)+`"`));return t}var _t=/[<>\r\n]/g;function vt(e){switch(e){case`<`:return`%3C`;case`>`:return`%3E`;case`
`:return`%0A`;case`\r`:return`%0D`;default:throw Error(`escapeLinkHrefForHeaderContextReplacer encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}}var yt=/["';,\r\n]/g;function W(e){switch(e){case`"`:return`%22`;case`'`:return`%27`;case`;`:return`%3B`;case`,`:return`%2C`;case`
`:return`%0A`;case`\r`:return`%0D`;default:throw Error(`escapeStringForLinkHeaderQuotedParamValueContextReplacer encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}}function G(e){this.styles.add(e)}function bt(e){this.stylesheets.add(e)}function xt(e,t){t.styles.forEach(G,e),t.stylesheets.forEach(bt,e),t.suspenseyImages&&(e.suspenseyImages=!0)}function St(e,t){var n=e.idPrefix,r=[],i=e.bootstrapScriptContent,a=e.bootstrapScripts,o=e.bootstrapModules;i!==void 0&&(r.push(`<script`),it(r,e),r.push(`>`,(``+i).replace(se,ce),`<\/script>`)),i=n+`P:`;var s=n+`S:`;n+=`B:`;var c=new Set,l=new Set,u=new Set,d=new Map,f=new Set,p=new Set,m=new Set,h={images:new Map,stylesheets:new Map,scripts:new Map,moduleScripts:new Map};if(a!==void 0)for(var g=0;g<a.length;g++){var _=a[g],v,y=void 0,b=void 0,x={rel:`preload`,as:`script`,fetchPriority:`low`,nonce:void 0};typeof _==`string`?x.href=v=_:(x.href=v=_.src,x.integrity=b=typeof _.integrity==`string`?_.integrity:void 0,x.crossOrigin=y=typeof _==`string`||_.crossOrigin==null?void 0:_.crossOrigin===`use-credentials`?`use-credentials`:``),_=e;var S=v;_.scriptResources[S]=null,_.moduleScriptResources[S]=null,_=[],Ee(_,x),f.add(_),r.push(`<script src="`,I(v),`"`),typeof b==`string`&&r.push(` integrity="`,I(b),`"`),typeof y==`string`&&r.push(` crossorigin="`,I(y),`"`),it(r,e),r.push(` async=""><\/script>`)}if(o!==void 0)for(a=0;a<o.length;a++)x=o[a],y=v=void 0,b={rel:`modulepreload`,fetchPriority:`low`,nonce:void 0},typeof x==`string`?b.href=g=x:(b.href=g=x.src,b.integrity=y=typeof x.integrity==`string`?x.integrity:void 0,b.crossOrigin=v=typeof x==`string`||x.crossOrigin==null?void 0:x.crossOrigin===`use-credentials`?`use-credentials`:``),x=e,_=g,x.scriptResources[_]=null,x.moduleScriptResources[_]=null,x=[],Ee(x,b),f.add(x),r.push(`<script type="module" src="`,I(g),`"`),typeof y==`string`&&r.push(` integrity="`,I(y),`"`),typeof v==`string`&&r.push(` crossorigin="`,I(v),`"`),it(r,e),r.push(` async=""><\/script>`);return{placeholderPrefix:i,segmentPrefix:s,boundaryPrefix:n,startInlineScript:`<script`,startInlineStyle:`<style`,preamble:{htmlChunks:null,headChunks:null,bodyChunks:null},externalRuntimeScript:null,bootstrapChunks:r,importMapChunks:[],onHeaders:void 0,headers:null,resets:{font:{},dns:{},connect:{default:{},anonymous:{},credentials:{}},image:{},style:{}},charsetChunks:[],viewportChunks:[],hoistableChunks:[],preconnects:c,fontPreloads:l,highImagePreloads:u,styles:d,bootstrapScripts:f,scripts:p,bulkPreloads:m,preloads:h,nonce:{script:void 0,style:void 0},stylesToHoist:!1,generateStaticMarkup:t}}function Ct(e,t,n,r){return n.generateStaticMarkup?(e.push(I(t)),!1):(t===``?e=r:(r&&e.push(`<!-- -->`),e.push(I(t)),e=!0),e)}function wt(e,t,n,r){t.generateStaticMarkup||n&&r&&e.push(`<!-- -->`)}var Tt=Function.prototype.bind,Et=Symbol.for(`react.client.reference`);function Dt(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===Et?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case c:return`Fragment`;case u:return`Profiler`;case l:return`StrictMode`;case m:return`Suspense`;case h:return`SuspenseList`;case y:return`Activity`}if(typeof e==`object`)switch(e.$$typeof){case s:return`Portal`;case f:return e.displayName||`Context`;case d:return(e._context.displayName||`Context`)+`.Consumer`;case p:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case g:return t=e.displayName||null,t===null?Dt(e.type)||`Memo`:t;case _:t=e._payload,e=e._init;try{return Dt(e(t))}catch{}}return null}var K={},Ot=null;function kt(e,t){if(e!==t){e.context._currentValue2=e.parentValue,e=e.parent;var n=t.parent;if(e===null){if(n!==null)throw Error(a(401))}else{if(n===null)throw Error(a(401));kt(e,n)}t.context._currentValue2=t.value}}function At(e){e.context._currentValue2=e.parentValue,e=e.parent,e!==null&&At(e)}function jt(e){var t=e.parent;t!==null&&jt(t),e.context._currentValue2=e.value}function Mt(e,t){if(e.context._currentValue2=e.parentValue,e=e.parent,e===null)throw Error(a(402));e.depth===t.depth?kt(e,t):Mt(e,t)}function Nt(e,t){var n=t.parent;if(n===null)throw Error(a(402));e.depth===n.depth?kt(e,n):Nt(e,n),t.context._currentValue2=t.value}function q(e){var t=Ot;t!==e&&(t===null?jt(e):e===null?At(t):t.depth===e.depth?kt(t,e):t.depth>e.depth?Mt(t,e):Nt(t,e),Ot=e)}var Pt={enqueueSetState:function(e,t){e=e._reactInternals,e.queue!==null&&e.queue.push(t)},enqueueReplaceState:function(e,t){e=e._reactInternals,e.replace=!0,e.queue=[t]},enqueueForceUpdate:function(){}},Ft={id:1,overflow:``};function J(e,t,n){var r=e.id;e=e.overflow;var i=32-It(r)-1;r&=~(1<<i),n+=1;var a=32-It(t)+i;if(30<a){var o=i-i%5;return a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,{id:1<<32-It(t)+i|n<<i|r,overflow:a+e}}return{id:1<<a|n<<i|r,overflow:e}}var It=Math.clz32?Math.clz32:zt,Lt=Math.log,Rt=Math.LN2;function zt(e){return e>>>=0,e===0?32:31-(Lt(e)/Rt|0)|0}function Y(){}var Bt=Error(a(460));function Vt(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(Y,Y),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw t.reason;default:switch(typeof t.status==`string`?t.then(Y,Y):(e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})),t.status){case`fulfilled`:return t.value;case`rejected`:throw t.reason}throw Ht=t,Bt}}var Ht=null;function Ut(){if(Ht===null)throw Error(a(459));var e=Ht;return Ht=null,e}function Wt(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var Gt=typeof Object.is==`function`?Object.is:Wt,Kt=null,qt=null,Jt=null,Yt=null,Xt=null,Zt=null,Qt=!1,$t=!1,en=0,tn=0,nn=-1,rn=0,an=null,on=null,sn=0;function cn(){if(Kt===null)throw Error(a(321));return Kt}function ln(){if(0<sn)throw Error(a(312));return{memoizedState:null,queue:null,next:null}}function un(){return Zt===null?Xt===null?(Qt=!1,Xt=Zt=ln()):(Qt=!0,Zt=Xt):Zt.next===null?(Qt=!1,Zt=Zt.next=ln()):(Qt=!0,Zt=Zt.next),Zt}function dn(){var e=an;return an=null,e}function fn(){Yt=Jt=qt=Kt=null,$t=!1,Xt=null,sn=0,Zt=on=null}function pn(e,t){return typeof t==`function`?t(e):t}function mn(e,t,n){if(Kt=cn(),Zt=un(),Qt){var r=Zt.queue;if(t=r.dispatch,on!==null&&(n=on.get(r),n!==void 0)){on.delete(r),r=Zt.memoizedState;do r=e(r,n.action),n=n.next;while(n!==null);return Zt.memoizedState=r,[r,t]}return[Zt.memoizedState,t]}return e=e===pn?typeof t==`function`?t():t:n===void 0?t:n(t),Zt.memoizedState=e,e=Zt.queue={last:null,dispatch:null},e=e.dispatch=gn.bind(null,Kt,e),[Zt.memoizedState,e]}function hn(e,t){if(Kt=cn(),Zt=un(),t=t===void 0?null:t,Zt!==null){var n=Zt.memoizedState;if(n!==null&&t!==null){var r=n[1];a:if(r===null)r=!1;else{for(var i=0;i<r.length&&i<t.length;i++)if(!Gt(t[i],r[i])){r=!1;break a}r=!0}if(r)return n[0]}}return e=e(),Zt.memoizedState=[e,t],e}function gn(e,t,n){if(25<=sn)throw Error(a(301));if(e===Kt)if($t=!0,e={action:n,next:null},on===null&&(on=new Map),n=on.get(t),n===void 0)on.set(t,e);else{for(t=n;t.next!==null;)t=t.next;t.next=e}}function _n(){throw Error(a(440))}function vn(){throw Error(a(394))}function yn(){throw Error(a(479))}function bn(e,t,n){cn();var r=tn++,i=Jt;if(typeof e.$$FORM_ACTION==`function`){var a=null,o=Yt;i=i.formState;var s=e.$$IS_SIGNATURE_EQUAL;if(i!==null&&typeof s==`function`){var c=i[1];s.call(e,i[2],i[3])&&(a=n===void 0?`k`+E(JSON.stringify([o,null,r]),0):`p`+n,c===a&&(nn=r,t=i[0]))}var l=e.bind(null,t);return e=function(e){l(e)},typeof l.$$FORM_ACTION==`function`&&(e.$$FORM_ACTION=function(e){e=l.$$FORM_ACTION(e),n!==void 0&&(n+=``,e.action=n);var t=e.data;return t&&(a===null&&(a=n===void 0?`k`+E(JSON.stringify([o,null,r]),0):`p`+n),t.append(`$ACTION_KEY`,a)),e}),[t,e,!1]}var u=e.bind(null,t);return[t,function(e){u(e)},!1]}function xn(e){var t=rn;return rn+=1,an===null&&(an=[]),Vt(an,e,t)}function Sn(){throw Error(a(393))}var Cn={readContext:function(e){return e._currentValue2},use:function(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return xn(e);if(e.$$typeof===f)return e._currentValue2}throw Error(a(438,String(e)))},useContext:function(e){return cn(),e._currentValue2},useMemo:hn,useReducer:mn,useRef:function(e){Kt=cn(),Zt=un();var t=Zt.memoizedState;return t===null?(e={current:e},Zt.memoizedState=e):t},useState:function(e){return mn(pn,e)},useInsertionEffect:Y,useLayoutEffect:Y,useCallback:function(e,t){return hn(function(){return e},t)},useImperativeHandle:Y,useEffect:Y,useDebugValue:Y,useDeferredValue:function(e,t){return cn(),t===void 0?e:t},useTransition:function(){return cn(),[!1,vn]},useId:function(){var e=qt.treeContext,t=e.overflow;e=e.id,e=(e&~(1<<32-It(e)-1)).toString(32)+t;var n=wn;if(n===null)throw Error(a(404));return t=en++,e=`_`+n.idPrefix+`R_`+e,0<t&&(e+=`H`+t.toString(32)),e+`_`},useSyncExternalStore:function(e,t,n){if(n===void 0)throw Error(a(407));return n()},useOptimistic:function(e){return cn(),[e,yn]},useActionState:bn,useFormState:bn,useHostTransitionStatus:function(){return cn(),ie},useMemoCache:function(e){for(var t=Array(e),n=0;n<e;n++)t[n]=x;return t},useCacheRefresh:function(){return Sn},useEffectEvent:function(){return _n}},wn=null,Tn={getCacheForType:function(){throw Error(a(248))},cacheSignal:function(){throw Error(a(248))}},En,Dn;function On(e){if(En===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);En=t&&t[1]||``,Dn=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+En+e+Dn}var kn=!1;function An(e,t){if(!e||kn)return``;kn=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,`props`,{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}e.call(n.prototype)}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,`name`,{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{kn=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?On(n):``}function jn(e){if(typeof e==`string`)return On(e);if(typeof e==`function`)return e.prototype&&e.prototype.isReactComponent?An(e,!0):An(e,!1);if(typeof e==`object`&&e){switch(e.$$typeof){case p:return An(e.render,!1);case g:return An(e.type,!1);case _:var t=e,n=t._payload;t=t._init;try{e=t(n)}catch{return On(`Lazy`)}return jn(e)}if(typeof e.name==`string`){a:{n=e.name,t=e.env;var r=e.debugLocation;if(r!=null&&(e=Error.prepareStackTrace,Error.prepareStackTrace=void 0,r=r.stack,Error.prepareStackTrace=e,r.startsWith(`Error: react-stack-top-frame
`)&&(r=r.slice(29)),e=r.indexOf(`
`),e!==-1&&(r=r.slice(e+1)),e=r.indexOf(`react_stack_bottom_frame`),e!==-1&&(e=r.lastIndexOf(`
`,e)),e=e===-1?``:r=r.slice(0,e),r=e.lastIndexOf(`
`),e=r===-1?e:e.slice(r+1),e.indexOf(n)!==-1)){n=`
`+e;break a}n=On(n+(t?` [`+t+`]`:``))}return n}}switch(e){case h:return On(`SuspenseList`);case m:return On(`Suspense`)}return``}function Mn(e,t){return(500<t.byteSize||!1)&&t.contentPreamble===null}function Nn(e){if(typeof e==`object`&&e&&typeof e.environmentName==`string`){var t=e.environmentName;e=[e].slice(0),typeof e[0]==`string`?e.splice(0,1,`[%s] `+e[0],` `+t+` `):e.splice(0,0,`[%s]`,` `+t+` `),e.unshift(console),t=Tt.apply(console.error,e),t()}else console.error(e);return null}function Pn(e,t,n,r,i,a,o,s,c,l,u){var d=new Set;this.destination=null,this.flushScheduled=!1,this.resumableState=e,this.renderState=t,this.rootFormatContext=n,this.progressiveChunkSize=r===void 0?12800:r,this.status=10,this.fatalError=null,this.pendingRootTasks=this.allPendingTasks=this.nextSegmentId=0,this.completedPreambleSegments=this.completedRootSegment=null,this.byteSize=0,this.abortableTasks=d,this.pingedTasks=[],this.clientRenderedBoundaries=[],this.completedBoundaries=[],this.partialBoundaries=[],this.trackedPostpones=null,this.onError=i===void 0?Nn:i,this.onPostpone=l===void 0?Y:l,this.onAllReady=a===void 0?Y:a,this.onShellReady=o===void 0?Y:o,this.onShellError=s===void 0?Y:s,this.onFatalError=c===void 0?Y:c,this.formState=u===void 0?null:u}function Fn(e,t,n,r,i,a,o,s,c,l,u,d){return t=new Pn(t,n,r,i,a,o,s,c,l,u,d),n=Vn(t,0,null,r,!1,!1),n.parentFlushed=!0,e=zn(t,null,e,-1,null,n,null,null,t.abortableTasks,null,r,null,Ft,null,null),Hn(e),t.pingedTasks.push(e),t}var In=null;function Ln(e,t){e.pingedTasks.push(t),e.pingedTasks.length===1&&(e.flushScheduled=e.destination!==null,yr(e))}function Rn(e,t,n,r,i){return n={status:0,rootSegmentID:-1,parentFlushed:!1,pendingTasks:0,row:t,completedSegments:[],byteSize:0,fallbackAbortableTasks:n,errorDigest:null,contentState:st(),fallbackState:st(),contentPreamble:r,fallbackPreamble:i,trackedContentKeyPath:null,trackedFallbackNode:null},t!==null&&(t.pendingTasks++,r=t.boundaries,r!==null&&(e.allPendingTasks++,n.pendingTasks++,r.push(n)),e=t.inheritedHoistables,e!==null&&xt(n.contentState,e)),n}function zn(e,t,n,r,i,a,o,s,c,l,u,d,f,p,m){e.allPendingTasks++,i===null?e.pendingRootTasks++:i.pendingTasks++,p!==null&&p.pendingTasks++;var h={replay:null,node:n,childIndex:r,ping:function(){return Ln(e,h)},blockedBoundary:i,blockedSegment:a,blockedPreamble:o,hoistableState:s,abortSet:c,keyPath:l,formatContext:u,context:d,treeContext:f,row:p,componentStack:m,thenableState:t};return c.add(h),h}function Bn(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.allPendingTasks++,a===null?e.pendingRootTasks++:a.pendingTasks++,f!==null&&f.pendingTasks++,n.pendingTasks++;var m={replay:n,node:r,childIndex:i,ping:function(){return Ln(e,m)},blockedBoundary:a,blockedSegment:null,blockedPreamble:null,hoistableState:o,abortSet:s,keyPath:c,formatContext:l,context:u,treeContext:d,row:f,componentStack:p,thenableState:t};return s.add(m),m}function Vn(e,t,n,r,i,a){return{status:0,parentFlushed:!1,id:-1,index:t,chunks:[],children:[],preambleChildren:[],parentFormatContext:r,boundary:n,lastPushedText:i,textEmbedded:a}}function Hn(e){var t=e.node;if(typeof t==`object`&&t)switch(t.$$typeof){case o:e.componentStack={parent:e.componentStack,type:t.type}}}function Un(e){return e===null?null:{parent:e.parent,type:`Suspense Fallback`}}function Wn(e){var t={};return e&&Object.defineProperty(t,`componentStack`,{configurable:!0,enumerable:!0,get:function(){try{var n=``,r=e;do n+=jn(r.type),r=r.parent;while(r);var i=n}catch(e){i=`
Error generating stack: `+e.message+`
`+e.stack}return Object.defineProperty(t,`componentStack`,{value:i}),i}}),t}function Gn(e,t,n){if(e=e.onError,t=e(t,n),t==null||typeof t==`string`)return t}function Kn(e,t){var n=e.onShellError,r=e.onFatalError;n(t),r(t),e.destination===null?(e.status=13,e.fatalError=t):(e.status=14,e.destination.destroy(t))}function qn(e,t){Jn(e,t.next,t.hoistables)}function Jn(e,t,n){for(;t!==null;){n!==null&&(xt(t.hoistables,n),t.inheritedHoistables=n);var r=t.boundaries;if(r!==null){t.boundaries=null;for(var i=0;i<r.length;i++){var a=r[i];n!==null&&xt(a.contentState,n),vr(e,a,null,null)}}if(t.pendingTasks--,0<t.pendingTasks)break;n=t.hoistables,t=t.next}}function Yn(e,t){var n=t.boundaries;if(n!==null&&t.pendingTasks===n.length){for(var r=!0,i=0;i<n.length;i++){var a=n[i];if(a.pendingTasks!==1||a.parentFlushed||Mn(e,a)){r=!1;break}}r&&Jn(e,t,t.hoistables)}}function Xn(e){var t={pendingTasks:1,boundaries:null,hoistables:st(),inheritedHoistables:null,together:!1,next:null};return e!==null&&0<e.pendingTasks&&(t.pendingTasks++,t.boundaries=[],e.next=t),t}function Zn(e,t,n,r,i){var a=t.keyPath,o=t.treeContext,s=t.row;t.keyPath=n,n=r.length;var c=null;if(t.replay!==null){var l=t.replay.slots;if(typeof l==`object`&&l)for(var u=0;u<n;u++){var d=i!==`backwards`&&i!==`unstable_legacy-backwards`?u:n-1-u,f=r[d];t.row=c=Xn(c),t.treeContext=J(o,n,d);var p=l[d];typeof p==`number`?(tr(e,t,p,f,d),delete l[d]):ur(e,t,f,d),--c.pendingTasks===0&&qn(e,c)}else for(l=0;l<n;l++)u=i!==`backwards`&&i!==`unstable_legacy-backwards`?l:n-1-l,d=r[u],t.row=c=Xn(c),t.treeContext=J(o,n,u),ur(e,t,d,u),--c.pendingTasks===0&&qn(e,c)}else if(i!==`backwards`&&i!==`unstable_legacy-backwards`)for(i=0;i<n;i++)l=r[i],t.row=c=Xn(c),t.treeContext=J(o,n,i),ur(e,t,l,i),--c.pendingTasks===0&&qn(e,c);else{for(i=t.blockedSegment,l=i.children.length,u=i.chunks.length,d=n-1;0<=d;d--){f=r[d],t.row=c=Xn(c),t.treeContext=J(o,n,d),p=Vn(e,u,null,t.formatContext,d===0?i.lastPushedText:!0,!0),i.children.splice(l,0,p),t.blockedSegment=p;try{ur(e,t,f,d),wt(p.chunks,e.renderState,p.lastPushedText,p.textEmbedded),p.status=1,--c.pendingTasks===0&&qn(e,c)}catch(t){throw p.status=e.status===12?3:4,t}}t.blockedSegment=i,i.lastPushedText=!1}s!==null&&c!==null&&0<c.pendingTasks&&(s.pendingTasks++,c.next=s),t.treeContext=o,t.row=s,t.keyPath=a}function Qn(e,t,n,r,i,a){var o=t.thenableState;for(t.thenableState=null,Kt={},qt=t,Jt=e,Yt=n,tn=en=0,nn=-1,rn=0,an=o,e=r(i,a);$t;)$t=!1,tn=en=0,nn=-1,rn=0,sn+=1,Zt=null,e=r(i,a);return fn(),e}function $n(e,t,n,r,i,a,o){var s=!1;if(a!==0&&e.formState!==null){var c=t.blockedSegment;if(c!==null){s=!0,c=c.chunks;for(var l=0;l<a;l++)l===o?c.push(`<!--F!-->`):c.push(`<!--F-->`)}}a=t.keyPath,t.keyPath=n,i?(n=t.treeContext,t.treeContext=J(n,1,0),ur(e,t,r,-1),t.treeContext=n):s?ur(e,t,r,-1):nr(e,t,r,-1),t.keyPath=a}function er(e,t,n,r,i,o){if(typeof r==`function`)if(r.prototype&&r.prototype.isReactComponent){var s=i;if(`ref`in i)for(var x in s={},i)x!==`ref`&&(s[x]=i[x]);var C=r.defaultProps;if(C)for(var E in s===i&&(s=D({},s,i)),C)s[E]===void 0&&(s[E]=C[E]);i=s,s=K,C=r.contextType,typeof C==`object`&&C&&(s=C._currentValue2),s=new r(i,s);var O=s.state===void 0?null:s.state;if(s.updater=Pt,s.props=i,s.state=O,C={queue:[],replace:!1},s._reactInternals=C,o=r.contextType,s.context=typeof o==`object`&&o?o._currentValue2:K,o=r.getDerivedStateFromProps,typeof o==`function`&&(o=o(i,O),O=o==null?O:D({},O,o),s.state=O),typeof r.getDerivedStateFromProps!=`function`&&typeof s.getSnapshotBeforeUpdate!=`function`&&(typeof s.UNSAFE_componentWillMount==`function`||typeof s.componentWillMount==`function`))if(r=s.state,typeof s.componentWillMount==`function`&&s.componentWillMount(),typeof s.UNSAFE_componentWillMount==`function`&&s.UNSAFE_componentWillMount(),r!==s.state&&Pt.enqueueReplaceState(s,s.state,null),C.queue!==null&&0<C.queue.length)if(r=C.queue,o=C.replace,C.queue=null,C.replace=!1,o&&r.length===1)s.state=r[0];else{for(C=o?r[0]:s.state,O=!0,o=o?1:0;o<r.length;o++)E=r[o],E=typeof E==`function`?E.call(s,C,i,void 0):E,E!=null&&(O?(O=!1,C=D({},C,E)):D(C,E));s.state=C}else C.queue=null;if(r=s.render(),e.status===12)throw null;i=t.keyPath,t.keyPath=n,nr(e,t,r,-1),t.keyPath=i}else{if(r=Qn(e,t,n,r,i,void 0),e.status===12)throw null;$n(e,t,n,r,en!==0,tn,nn)}else if(typeof r==`string`)if(s=t.blockedSegment,s===null)s=i.children,C=t.formatContext,O=t.keyPath,t.formatContext=ue(C,r,i),t.keyPath=n,ur(e,t,s,-1),t.formatContext=C,t.keyPath=O;else{if(O=Fe(s.chunks,r,i,e.resumableState,e.renderState,t.blockedPreamble,t.hoistableState,t.formatContext,s.lastPushedText),s.lastPushedText=!1,C=t.formatContext,o=t.keyPath,t.keyPath=n,(t.formatContext=ue(C,r,i)).insertionMode===3){n=Vn(e,0,null,t.formatContext,!1,!1),s.preambleChildren.push(n),t.blockedSegment=n;try{n.status=6,ur(e,t,O,-1),wt(n.chunks,e.renderState,n.lastPushedText,n.textEmbedded),n.status=1}finally{t.blockedSegment=s}}else ur(e,t,O,-1);t.formatContext=C,t.keyPath=o;a:{switch(t=s.chunks,e=e.resumableState,r){case`title`:case`style`:case`script`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`img`:case`input`:case`keygen`:case`link`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:break a;case`body`:if(1>=C.insertionMode){e.hasBody=!0;break a}break;case`html`:if(C.insertionMode===0){e.hasHtml=!0;break a}break;case`head`:if(1>=C.insertionMode)break a}t.push(Le(r))}s.lastPushedText=!1}else{switch(r){case b:case l:case u:case c:r=t.keyPath,t.keyPath=n,nr(e,t,i.children,-1),t.keyPath=r;return;case y:r=t.blockedSegment,r===null?i.mode!==`hidden`&&(r=t.keyPath,t.keyPath=n,ur(e,t,i.children,-1),t.keyPath=r):i.mode!==`hidden`&&(e.renderState.generateStaticMarkup||r.chunks.push(`<!--&-->`),r.lastPushedText=!1,s=t.keyPath,t.keyPath=n,ur(e,t,i.children,-1),t.keyPath=s,e.renderState.generateStaticMarkup||r.chunks.push(`<!--/&-->`),r.lastPushedText=!1);return;case h:a:{if(r=i.children,i=i.revealOrder,i===`forwards`||i===`backwards`||i===`unstable_legacy-backwards`){if(T(r)){Zn(e,t,n,r,i);break a}if((s=w(r))&&(s=s.call(r))){if(C=s.next(),!C.done){do C=s.next();while(!C.done);Zn(e,t,n,r,i)}break a}}i===`together`?(i=t.keyPath,s=t.row,C=t.row=Xn(null),C.boundaries=[],C.together=!0,t.keyPath=n,nr(e,t,r,-1),--C.pendingTasks===0&&qn(e,C),t.keyPath=i,t.row=s,s!==null&&0<C.pendingTasks&&(s.pendingTasks++,C.next=s)):(i=t.keyPath,t.keyPath=n,nr(e,t,r,-1),t.keyPath=i)}return;case S:case v:throw Error(a(343));case m:a:if(t.replay!==null){r=t.keyPath,s=t.formatContext,C=t.row,t.keyPath=n,t.formatContext=fe(e.resumableState,s),t.row=null,n=i.children;try{ur(e,t,n,-1)}finally{t.keyPath=r,t.formatContext=s,t.row=C}}else{r=t.keyPath,o=t.formatContext;var k=t.row,A=t.blockedBoundary;E=t.blockedPreamble;var j=t.hoistableState;x=t.blockedSegment;var M=i.fallback;i=i.children;var N=new Set,P=Rn(e,t.row,N,null,null);e.trackedPostpones!==null&&(P.trackedContentKeyPath=n);var F=Vn(e,x.chunks.length,P,t.formatContext,!1,!1);x.children.push(F),x.lastPushedText=!1;var I=Vn(e,0,null,t.formatContext,!1,!1);if(I.parentFlushed=!0,e.trackedPostpones!==null){s=t.componentStack,C=[n[0],`Suspense Fallback`,n[2]],O=[C[1],C[2],[],null],e.trackedPostpones.workingMap.set(C,O),P.trackedFallbackNode=O,t.blockedSegment=F,t.blockedPreamble=P.fallbackPreamble,t.keyPath=C,t.formatContext=de(e.resumableState,o),t.componentStack=Un(s),F.status=6;try{ur(e,t,M,-1),wt(F.chunks,e.renderState,F.lastPushedText,F.textEmbedded),F.status=1}catch(t){throw F.status=e.status===12?3:4,t}finally{t.blockedSegment=x,t.blockedPreamble=E,t.keyPath=r,t.formatContext=o}t=zn(e,null,i,-1,P,I,P.contentPreamble,P.contentState,t.abortSet,n,fe(e.resumableState,t.formatContext),t.context,t.treeContext,null,s),Hn(t),e.pingedTasks.push(t)}else{t.blockedBoundary=P,t.blockedPreamble=P.contentPreamble,t.hoistableState=P.contentState,t.blockedSegment=I,t.keyPath=n,t.formatContext=fe(e.resumableState,o),t.row=null,I.status=6;try{if(ur(e,t,i,-1),wt(I.chunks,e.renderState,I.lastPushedText,I.textEmbedded),I.status=1,_r(P,I),P.pendingTasks===0&&P.status===0){if(P.status=1,!Mn(e,P)){k!==null&&--k.pendingTasks===0&&qn(e,k),e.pendingRootTasks===0&&t.blockedPreamble&&Sr(e);break a}}else k!==null&&k.together&&Yn(e,k)}catch(n){P.status=4,e.status===12?(I.status=3,s=e.fatalError):(I.status=4,s=n),C=Wn(t.componentStack),O=Gn(e,s,C),P.errorDigest=O,sr(e,P)}finally{t.blockedBoundary=A,t.blockedPreamble=E,t.hoistableState=j,t.blockedSegment=x,t.keyPath=r,t.formatContext=o,t.row=k}t=zn(e,null,M,-1,A,F,P.fallbackPreamble,P.fallbackState,N,[n[0],`Suspense Fallback`,n[2]],de(e.resumableState,t.formatContext),t.context,t.treeContext,t.row,Un(t.componentStack)),Hn(t),e.pingedTasks.push(t)}}return}if(typeof r==`object`&&r)switch(r.$$typeof){case p:if(`ref`in i)for(M in s={},i)M!==`ref`&&(s[M]=i[M]);else s=i;r=Qn(e,t,n,r.render,s,o),$n(e,t,n,r,en!==0,tn,nn);return;case g:er(e,t,n,r.type,i,o);return;case f:if(C=i.children,s=t.keyPath,i=i.value,O=r._currentValue2,r._currentValue2=i,o=Ot,Ot=r={parent:o,depth:o===null?0:o.depth+1,context:r,parentValue:O,value:i},t.context=r,t.keyPath=n,nr(e,t,C,-1),e=Ot,e===null)throw Error(a(403));e.context._currentValue2=e.parentValue,e=Ot=e.parent,t.context=e,t.keyPath=s;return;case d:i=i.children,r=i(r._context._currentValue2),i=t.keyPath,t.keyPath=n,nr(e,t,r,-1),t.keyPath=i;return;case _:if(s=r._init,r=s(r._payload),e.status===12)throw null;er(e,t,n,r,i,o);return}throw Error(a(130,r==null?r:typeof r,``))}}function tr(e,t,n,r,i){var a=t.replay,o=t.blockedBoundary,s=Vn(e,0,null,t.formatContext,!1,!1);s.id=n,s.parentFlushed=!0;try{t.replay=null,t.blockedSegment=s,ur(e,t,r,i),s.status=1,o===null?e.completedRootSegment=s:(_r(o,s),o.parentFlushed&&e.partialBoundaries.push(o))}finally{t.replay=a,t.blockedSegment=null}}function nr(e,t,n,r){t.replay!==null&&typeof t.replay.slots==`number`?tr(e,t,t.replay.slots,n,r):(t.node=n,t.childIndex=r,n=t.componentStack,Hn(t),rr(e,t),t.componentStack=n)}function rr(e,t){var n=t.node,r=t.childIndex;if(n!==null){if(typeof n==`object`){switch(n.$$typeof){case o:var i=n.type,c=n.key,l=n.props;n=l.ref;var u=n===void 0?null:n,d=Dt(i),p=c??(r===-1?0:r);if(c=[t.keyPath,d,p],t.replay!==null)a:{var h=t.replay;for(r=h.nodes,n=0;n<r.length;n++){var g=r[n];if(p===g[1]){if(g.length===4){if(d!==null&&d!==g[0])throw Error(a(490,g[0],d));var v=g[2];d=g[3],p=t.node,t.replay={nodes:v,slots:d,pendingTasks:1};try{if(er(e,t,c,i,l,u),t.replay.pendingTasks===1&&0<t.replay.nodes.length)throw Error(a(488));t.replay.pendingTasks--}catch(a){if(typeof a==`object`&&a&&(a===Bt||typeof a.then==`function`))throw t.node===p?t.replay=h:r.splice(n,1),a;t.replay.pendingTasks--,l=Wn(t.componentStack),c=e,e=t.blockedBoundary,i=a,l=Gn(c,i,l),fr(c,e,v,d,i,l)}t.replay=h}else{if(i!==m)throw Error(a(490,`Suspense`,Dt(i)||`Unknown`));b:{h=void 0,i=g[5],u=g[2],d=g[3],p=g[4]===null?[]:g[4][2],g=g[4]===null?null:g[4][3];var y=t.keyPath,b=t.formatContext,x=t.row,S=t.replay,C=t.blockedBoundary,E=t.hoistableState,D=l.children,O=l.fallback,k=new Set;l=Rn(e,t.row,k,null,null),l.parentFlushed=!0,l.rootSegmentID=i,t.blockedBoundary=l,t.hoistableState=l.contentState,t.keyPath=c,t.formatContext=fe(e.resumableState,b),t.row=null,t.replay={nodes:u,slots:d,pendingTasks:1};try{if(ur(e,t,D,-1),t.replay.pendingTasks===1&&0<t.replay.nodes.length)throw Error(a(488));if(t.replay.pendingTasks--,l.pendingTasks===0&&l.status===0){l.status=1,e.completedBoundaries.push(l);break b}}catch(n){l.status=4,v=Wn(t.componentStack),h=Gn(e,n,v),l.errorDigest=h,t.replay.pendingTasks--,e.clientRenderedBoundaries.push(l)}finally{t.blockedBoundary=C,t.hoistableState=E,t.replay=S,t.keyPath=y,t.formatContext=b,t.row=x}v=Bn(e,null,{nodes:p,slots:g,pendingTasks:0},O,-1,C,l.fallbackState,k,[c[0],`Suspense Fallback`,c[2]],de(e.resumableState,t.formatContext),t.context,t.treeContext,t.row,Un(t.componentStack)),Hn(v),e.pingedTasks.push(v)}}r.splice(n,1);break a}}}else er(e,t,c,i,l,u);return;case s:throw Error(a(257));case _:if(v=n._init,n=v(n._payload),e.status===12)throw null;nr(e,t,n,r);return}if(T(n)){ir(e,t,n,r);return}if((v=w(n))&&(v=v.call(n))){if(n=v.next(),!n.done){l=[];do l.push(n.value),n=v.next();while(!n.done);ir(e,t,l,r)}return}if(typeof n.then==`function`)return t.thenableState=null,nr(e,t,xn(n),r);if(n.$$typeof===f)return nr(e,t,n._currentValue2,r);throw r=Object.prototype.toString.call(n),Error(a(31,r===`[object Object]`?`object with keys {`+Object.keys(n).join(`, `)+`}`:r))}typeof n==`string`?(r=t.blockedSegment,r!==null&&(r.lastPushedText=Ct(r.chunks,n,e.renderState,r.lastPushedText))):(typeof n==`number`||typeof n==`bigint`)&&(r=t.blockedSegment,r!==null&&(r.lastPushedText=Ct(r.chunks,``+n,e.renderState,r.lastPushedText)))}}function ir(e,t,n,r){var i=t.keyPath;if(r!==-1&&(t.keyPath=[t.keyPath,`Fragment`,r],t.replay!==null)){for(var o=t.replay,s=o.nodes,c=0;c<s.length;c++){var l=s[c];if(l[1]===r){r=l[2],l=l[3],t.replay={nodes:r,slots:l,pendingTasks:1};try{if(ir(e,t,n,-1),t.replay.pendingTasks===1&&0<t.replay.nodes.length)throw Error(a(488));t.replay.pendingTasks--}catch(i){if(typeof i==`object`&&i&&(i===Bt||typeof i.then==`function`))throw i;t.replay.pendingTasks--,n=Wn(t.componentStack);var u=t.blockedBoundary,d=i;n=Gn(e,d,n),fr(e,u,r,l,d,n)}t.replay=o,s.splice(c,1);break}}t.keyPath=i;return}if(o=t.treeContext,s=n.length,t.replay!==null&&(c=t.replay.slots,typeof c==`object`&&c)){for(r=0;r<s;r++)l=n[r],t.treeContext=J(o,s,r),u=c[r],typeof u==`number`?(tr(e,t,u,l,r),delete c[r]):ur(e,t,l,r);t.treeContext=o,t.keyPath=i;return}for(c=0;c<s;c++)r=n[c],t.treeContext=J(o,s,c),ur(e,t,r,c);t.treeContext=o,t.keyPath=i}function ar(e,t,n){if(n.status=5,n.rootSegmentID=e.nextSegmentId++,e=n.trackedContentKeyPath,e===null)throw Error(a(486));var r=n.trackedFallbackNode,i=[],o=t.workingMap.get(e);return o===void 0?(n=[e[1],e[2],i,null,r,n.rootSegmentID],t.workingMap.set(e,n),Pr(n,e[0],t),n):(o[4]=r,o[5]=n.rootSegmentID,o)}function or(e,t,n,r){r.status=5;var i=n.keyPath,o=n.blockedBoundary;if(o===null)r.id=e.nextSegmentId++,t.rootSlots=r.id,e.completedRootSegment!==null&&(e.completedRootSegment.status=5);else{if(o!==null&&o.status===0){var s=ar(e,t,o);if(o.trackedContentKeyPath===i&&n.childIndex===-1){r.id===-1&&(r.id=r.parentFlushed?o.rootSegmentID:e.nextSegmentId++),s[3]=r.id;return}}if(r.id===-1&&(r.id=r.parentFlushed&&o!==null?o.rootSegmentID:e.nextSegmentId++),n.childIndex===-1)i===null?t.rootSlots=r.id:(n=t.workingMap.get(i),n===void 0?(n=[i[1],i[2],[],r.id],Pr(n,i[0],t)):n[3]=r.id);else{if(i===null){if(e=t.rootSlots,e===null)e=t.rootSlots={};else if(typeof e==`number`)throw Error(a(491))}else if(o=t.workingMap,s=o.get(i),s===void 0)e={},s=[i[1],i[2],[],e],o.set(i,s),Pr(s,i[0],t);else if(e=s[3],e===null)e=s[3]={};else if(typeof e==`number`)throw Error(a(491));e[n.childIndex]=r.id}}}function sr(e,t){e=e.trackedPostpones,e!==null&&(t=t.trackedContentKeyPath,t!==null&&(t=e.workingMap.get(t),t!==void 0&&(t.length=4,t[2]=[],t[3]=null)))}function cr(e,t,n){return Bn(e,n,t.replay,t.node,t.childIndex,t.blockedBoundary,t.hoistableState,t.abortSet,t.keyPath,t.formatContext,t.context,t.treeContext,t.row,t.componentStack)}function lr(e,t,n){var r=t.blockedSegment,i=Vn(e,r.chunks.length,null,t.formatContext,r.lastPushedText,!0);return r.children.push(i),r.lastPushedText=!1,zn(e,n,t.node,t.childIndex,t.blockedBoundary,i,t.blockedPreamble,t.hoistableState,t.abortSet,t.keyPath,t.formatContext,t.context,t.treeContext,t.row,t.componentStack)}function ur(e,t,n,r){var i=t.formatContext,a=t.context,o=t.keyPath,s=t.treeContext,c=t.componentStack,l=t.blockedSegment;if(l===null){l=t.replay;try{return nr(e,t,n,r)}catch(u){if(fn(),n=u===Bt?Ut():u,e.status!==12&&typeof n==`object`&&n){if(typeof n.then==`function`){r=u===Bt?dn():null,e=cr(e,t,r).ping,n.then(e,e),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,t.replay=l,q(a);return}if(n.message===`Maximum call stack size exceeded`){n=u===Bt?dn():null,n=cr(e,t,n),e.pingedTasks.push(n),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,t.replay=l,q(a);return}}}}else{var u=l.children.length,d=l.chunks.length;try{return nr(e,t,n,r)}catch(r){if(fn(),l.children.length=u,l.chunks.length=d,n=r===Bt?Ut():r,e.status!==12&&typeof n==`object`&&n){if(typeof n.then==`function`){l=n,n=r===Bt?dn():null,e=lr(e,t,n).ping,l.then(e,e),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,q(a);return}if(n.message===`Maximum call stack size exceeded`){l=r===Bt?dn():null,l=lr(e,t,l),e.pingedTasks.push(l),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,q(a);return}}}}throw t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,q(a),n}function dr(e){var t=e.blockedBoundary,n=e.blockedSegment;n!==null&&(n.status=3,vr(this,t,e.row,n))}function fr(e,t,n,r,i,o){for(var s=0;s<n.length;s++){var c=n[s];if(c.length===4)fr(e,t,c[2],c[3],i,o);else{c=c[5];var l=e,u=o,d=Rn(l,null,new Set,null,null);d.parentFlushed=!0,d.rootSegmentID=c,d.status=4,d.errorDigest=u,d.parentFlushed&&l.clientRenderedBoundaries.push(d)}}if(n.length=0,r!==null){if(t===null)throw Error(a(487));if(t.status!==4&&(t.status=4,t.errorDigest=o,t.parentFlushed&&e.clientRenderedBoundaries.push(t)),typeof r==`object`)for(var f in r)delete r[f]}}function pr(e,t,n){var r=e.blockedBoundary,i=e.blockedSegment;if(i!==null){if(i.status===6)return;i.status=3}var a=Wn(e.componentStack);if(r===null){if(t.status!==13&&t.status!==14){if(r=e.replay,r===null){t.trackedPostpones!==null&&i!==null?(r=t.trackedPostpones,Gn(t,n,a),or(t,r,e,i),vr(t,null,e.row,i)):(Gn(t,n,a),Kn(t,n));return}r.pendingTasks--,r.pendingTasks===0&&0<r.nodes.length&&(i=Gn(t,n,a),fr(t,null,r.nodes,r.slots,n,i)),t.pendingRootTasks--,t.pendingRootTasks===0&&hr(t)}}else{var o=t.trackedPostpones;if(r.status!==4){if(o!==null&&i!==null)return Gn(t,n,a),or(t,o,e,i),r.fallbackAbortableTasks.forEach(function(e){return pr(e,t,n)}),r.fallbackAbortableTasks.clear(),vr(t,r,e.row,i);r.status=4,i=Gn(t,n,a),r.status=4,r.errorDigest=i,sr(t,r),r.parentFlushed&&t.clientRenderedBoundaries.push(r)}r.pendingTasks--,i=r.row,i!==null&&--i.pendingTasks===0&&qn(t,i),r.fallbackAbortableTasks.forEach(function(e){return pr(e,t,n)}),r.fallbackAbortableTasks.clear()}e=e.row,e!==null&&--e.pendingTasks===0&&qn(t,e),t.allPendingTasks--,t.allPendingTasks===0&&gr(t)}function mr(e,t){try{var n=e.renderState,r=n.onHeaders;if(r){var i=n.headers;if(i){n.headers=null;var a=i.preconnects;if(i.fontPreloads&&(a&&(a+=`, `),a+=i.fontPreloads),i.highImagePreloads&&(a&&(a+=`, `),a+=i.highImagePreloads),!t){var o=n.styles.values(),s=o.next();b:for(;0<i.remainingCapacity&&!s.done;s=o.next())for(var c=s.value.sheets.values(),l=c.next();0<i.remainingCapacity&&!l.done;l=c.next()){var u=l.value,d=u.props,f=d.href,p=u.props,m=gt(p.href,`style`,{crossOrigin:p.crossOrigin,integrity:p.integrity,nonce:p.nonce,type:p.type,fetchPriority:p.fetchPriority,referrerPolicy:p.referrerPolicy,media:p.media});if(0<=(i.remainingCapacity-=m.length+2))n.resets.style[f]=ae,a&&(a+=`, `),a+=m,n.resets.style[f]=typeof d.crossOrigin==`string`||typeof d.integrity==`string`?[d.crossOrigin,d.integrity]:ae;else break b}}r(a?{Link:a}:{})}}}catch(t){Gn(e,t,{})}}function hr(e){e.trackedPostpones===null&&mr(e,!0),e.trackedPostpones===null&&Sr(e),e.onShellError=Y,e=e.onShellReady,e()}function gr(e){mr(e,e.trackedPostpones===null?!0:e.completedRootSegment===null||e.completedRootSegment.status!==5),Sr(e),e=e.onAllReady,e()}function _r(e,t){if(t.chunks.length===0&&t.children.length===1&&t.children[0].boundary===null&&t.children[0].id===-1){var n=t.children[0];n.id=t.id,n.parentFlushed=!0,n.status!==1&&n.status!==3&&n.status!==4||_r(e,n)}else e.completedSegments.push(t)}function vr(e,t,n,r){if(n!==null&&(--n.pendingTasks===0?qn(e,n):n.together&&Yn(e,n)),e.allPendingTasks--,t===null){if(r!==null&&r.parentFlushed){if(e.completedRootSegment!==null)throw Error(a(389));e.completedRootSegment=r}e.pendingRootTasks--,e.pendingRootTasks===0&&hr(e)}else if(t.pendingTasks--,t.status!==4)if(t.pendingTasks===0){if(t.status===0&&(t.status=1),r!==null&&r.parentFlushed&&(r.status===1||r.status===3)&&_r(t,r),t.parentFlushed&&e.completedBoundaries.push(t),t.status===1)n=t.row,n!==null&&xt(n.hoistables,t.contentState),Mn(e,t)||(t.fallbackAbortableTasks.forEach(dr,e),t.fallbackAbortableTasks.clear(),n!==null&&--n.pendingTasks===0&&qn(e,n)),e.pendingRootTasks===0&&e.trackedPostpones===null&&t.contentPreamble!==null&&Sr(e);else if(t.status===5&&(t=t.row,t!==null)){if(e.trackedPostpones!==null){n=e.trackedPostpones;var i=t.next;if(i!==null&&(r=i.boundaries,r!==null))for(i.boundaries=null,i=0;i<r.length;i++){var o=r[i];ar(e,n,o),vr(e,o,null,null)}}--t.pendingTasks===0&&qn(e,t)}}else r===null||!r.parentFlushed||r.status!==1&&r.status!==3||(_r(t,r),t.completedSegments.length===1&&t.parentFlushed&&e.partialBoundaries.push(t)),t=t.row,t!==null&&t.together&&Yn(e,t);e.allPendingTasks===0&&gr(e)}function yr(e){if(e.status!==14&&e.status!==13){var t=Ot,n=ne.H;ne.H=Cn;var r=ne.A;ne.A=Tn;var i=In;In=e;var o=wn;wn=e.resumableState;try{var s=e.pingedTasks,c;for(c=0;c<s.length;c++){var l=s[c],u=e,d=l.blockedSegment;if(d===null){var f=u;if(l.replay.pendingTasks!==0){q(l.context);try{if(typeof l.replay.slots==`number`?tr(f,l,l.replay.slots,l.node,l.childIndex):rr(f,l),l.replay.pendingTasks===1&&0<l.replay.nodes.length)throw Error(a(488));l.replay.pendingTasks--,l.abortSet.delete(l),vr(f,l.blockedBoundary,l.row,null)}catch(e){fn();var p=e===Bt?Ut():e;if(typeof p==`object`&&p&&typeof p.then==`function`){var m=l.ping;p.then(m,m),l.thenableState=e===Bt?dn():null}else{l.replay.pendingTasks--,l.abortSet.delete(l);var h=Wn(l.componentStack);u=void 0;var g=f,_=l.blockedBoundary,v=f.status===12?f.fatalError:p,y=l.replay.nodes,b=l.replay.slots;u=Gn(g,v,h),fr(g,_,y,b,v,u),f.pendingRootTasks--,f.pendingRootTasks===0&&hr(f),f.allPendingTasks--,f.allPendingTasks===0&&gr(f)}}}}else if(f=void 0,g=d,g.status===0){g.status=6,q(l.context);var x=g.children.length,S=g.chunks.length;try{rr(u,l),wt(g.chunks,u.renderState,g.lastPushedText,g.textEmbedded),l.abortSet.delete(l),g.status=1,vr(u,l.blockedBoundary,l.row,g)}catch(e){fn(),g.children.length=x,g.chunks.length=S;var C=e===Bt?Ut():u.status===12?u.fatalError:e;if(u.status===12&&u.trackedPostpones!==null){var w=u.trackedPostpones,T=Wn(l.componentStack);l.abortSet.delete(l),Gn(u,C,T),or(u,w,l,g),vr(u,l.blockedBoundary,l.row,g)}else if(typeof C==`object`&&C&&typeof C.then==`function`){g.status=0,l.thenableState=e===Bt?dn():null;var E=l.ping;C.then(E,E)}else{var D=Wn(l.componentStack);l.abortSet.delete(l),g.status=4;var O=l.blockedBoundary,k=l.row;if(k!==null&&--k.pendingTasks===0&&qn(u,k),u.allPendingTasks--,f=Gn(u,C,D),O===null)Kn(u,C);else if(O.pendingTasks--,O.status!==4){O.status=4,O.errorDigest=f,sr(u,O);var A=O.row;A!==null&&--A.pendingTasks===0&&qn(u,A),O.parentFlushed&&u.clientRenderedBoundaries.push(O),u.pendingRootTasks===0&&u.trackedPostpones===null&&O.contentPreamble!==null&&Sr(u)}u.allPendingTasks===0&&gr(u)}}}}s.splice(0,c),e.destination!==null&&Ar(e,e.destination)}catch(t){Gn(e,t,{}),Kn(e,t)}finally{wn=o,ne.H=n,ne.A=r,n===Cn&&q(t),In=i}}}function br(e,t,n){t.preambleChildren.length&&n.push(t.preambleChildren);for(var r=!1,i=0;i<t.children.length;i++)r=xr(e,t.children[i],n)||r;return r}function xr(e,t,n){var r=t.boundary;if(r===null)return br(e,t,n);var i=r.contentPreamble,o=r.fallbackPreamble;if(i===null||o===null)return!1;switch(r.status){case 1:if(Re(e.renderState,i),e.byteSize+=r.byteSize,t=r.completedSegments[0],!t)throw Error(a(391));return br(e,t,n);case 5:if(e.trackedPostpones!==null)return!0;case 4:if(t.status===1)return Re(e.renderState,o),br(e,t,n);default:return!0}}function Sr(e){if(e.completedRootSegment&&e.completedPreambleSegments===null){var t=[],n=e.byteSize,r=xr(e,e.completedRootSegment,t),i=e.renderState.preamble;!1===r||i.headChunks&&i.bodyChunks?e.completedPreambleSegments=t:e.byteSize=n}}function Cr(e,t,n,r){switch(n.parentFlushed=!0,n.status){case 0:n.id=e.nextSegmentId++;case 5:return r=n.id,n.lastPushedText=!1,n.textEmbedded=!1,e=e.renderState,t.push(`<template id="`),t.push(e.placeholderPrefix),e=r.toString(16),t.push(e),t.push(`"></template>`);case 1:n.status=2;var i=!0,o=n.chunks,s=0;n=n.children;for(var c=0;c<n.length;c++){for(i=n[c];s<i.index;s++)t.push(o[s]);i=Tr(e,t,i,r)}for(;s<o.length-1;s++)t.push(o[s]);return s<o.length&&(i=t.push(o[s])),i;case 3:return!0;default:throw Error(a(390))}}var wr=0;function Tr(e,t,n,r){var i=n.boundary;if(i===null)return Cr(e,t,n,r);if(i.parentFlushed=!0,i.status===4){var o=i.row;return o!==null&&--o.pendingTasks===0&&qn(e,o),e.renderState.generateStaticMarkup||(i=i.errorDigest,t.push(`<!--$!-->`),t.push(`<template`),i&&(t.push(` data-dgst="`),i=I(i),t.push(i),t.push(`"`)),t.push(`></template>`)),Cr(e,t,n,r),e=e.renderState.generateStaticMarkup?!0:t.push(`<!--/$-->`),e}if(i.status!==1)return i.status===0&&(i.rootSegmentID=e.nextSegmentId++),0<i.completedSegments.length&&e.partialBoundaries.push(i),Be(t,e.renderState,i.rootSegmentID),r&&xt(r,i.fallbackState),Cr(e,t,n,r),t.push(`<!--/$-->`);if(!kr&&Mn(e,i)&&wr+i.byteSize>e.progressiveChunkSize)return i.rootSegmentID=e.nextSegmentId++,e.completedBoundaries.push(i),Be(t,e.renderState,i.rootSegmentID),Cr(e,t,n,r),t.push(`<!--/$-->`);if(wr+=i.byteSize,r&&xt(r,i.contentState),n=i.row,n!==null&&Mn(e,i)&&--n.pendingTasks===0&&qn(e,n),e.renderState.generateStaticMarkup||t.push(`<!--$-->`),n=i.completedSegments,n.length!==1)throw Error(a(391));return Tr(e,t,n[0],r),e=e.renderState.generateStaticMarkup?!0:t.push(`<!--/$-->`),e}function Er(e,t,n,r){return Ve(t,e.renderState,n.parentFormatContext,n.id),Tr(e,t,n,r),He(t,n.parentFormatContext)}function Dr(e,t,n){wr=n.byteSize;for(var r=n.completedSegments,i=0;i<r.length;i++)Or(e,t,n,r[i]);r.length=0,r=n.row,r!==null&&Mn(e,n)&&--r.pendingTasks===0&&qn(e,r),Ze(t,n.contentState,e.renderState),r=e.resumableState,e=e.renderState,i=n.rootSegmentID,n=n.contentState;var a=e.stylesToHoist;return e.stylesToHoist=!1,t.push(e.startInlineScript),t.push(`>`),a?(!(r.instructions&4)&&(r.instructions|=4,t.push(`$RX=function(b,c,d,e,f){var a=document.getElementById(b);a&&(b=a.previousSibling,b.data="$!",a=a.dataset,c&&(a.dgst=c),d&&(a.msg=d),e&&(a.stck=e),f&&(a.cstck=f),b._reactRetry&&b._reactRetry())};`)),!(r.instructions&2)&&(r.instructions|=2,t.push(`$RB=[];$RV=function(a){$RT=performance.now();for(var b=0;b<a.length;b+=2){var c=a[b],e=a[b+1];null!==e.parentNode&&e.parentNode.removeChild(e);var f=c.parentNode;if(f){var g=c.previousSibling,h=0;do{if(c&&8===c.nodeType){var d=c.data;if("/$"===d||"/&"===d)if(0===h)break;else h--;else"$"!==d&&"$?"!==d&&"$~"!==d&&"$!"!==d&&"&"!==d||h++}d=c.nextSibling;f.removeChild(c);c=d}while(c);for(;e.firstChild;)f.insertBefore(e.firstChild,c);g.data="$";g._reactRetry&&requestAnimationFrame(g._reactRetry)}}a.length=0};
$RC=function(a,b){if(b=document.getElementById(b))(a=document.getElementById(a))?(a.previousSibling.data="$~",$RB.push(a,b),2===$RB.length&&("number"!==typeof $RT?requestAnimationFrame($RV.bind(null,$RB)):(a=performance.now(),setTimeout($RV.bind(null,$RB),2300>a&&2E3<a?2300-a:$RT+300-a)))):b.parentNode.removeChild(b)};`)),r.instructions&8?t.push(`$RR("`):(r.instructions|=8,t.push(`$RM=new Map;$RR=function(n,w,p){function u(q){this._p=null;q()}for(var r=new Map,t=document,h,b,e=t.querySelectorAll("link[data-precedence],style[data-precedence]"),v=[],k=0;b=e[k++];)"not all"===b.getAttribute("media")?v.push(b):("LINK"===b.tagName&&$RM.set(b.getAttribute("href"),b),r.set(b.dataset.precedence,h=b));e=0;b=[];var l,a;for(k=!0;;){if(k){var f=p[e++];if(!f){k=!1;e=0;continue}var c=!1,m=0;var d=f[m++];if(a=$RM.get(d)){var g=a._p;c=!0}else{a=t.createElement("link");a.href=d;a.rel=
"stylesheet";for(a.dataset.precedence=l=f[m++];g=f[m++];)a.setAttribute(g,f[m++]);g=a._p=new Promise(function(q,x){a.onload=u.bind(a,q);a.onerror=u.bind(a,x)});$RM.set(d,a)}d=a.getAttribute("media");!g||d&&!matchMedia(d).matches||b.push(g);if(c)continue}else{a=v[e++];if(!a)break;l=a.getAttribute("data-precedence");a.removeAttribute("media")}c=r.get(l)||h;c===h&&(h=a);r.set(l,a);c?c.parentNode.insertBefore(a,c.nextSibling):(c=t.head,c.insertBefore(a,c.firstChild))}if(p=document.getElementById(n))p.previousSibling.data=
"$~";Promise.all(b).then($RC.bind(null,n,w),$RX.bind(null,n,"CSS failed to load"))};$RR("`))):(!(r.instructions&2)&&(r.instructions|=2,t.push(`$RB=[];$RV=function(a){$RT=performance.now();for(var b=0;b<a.length;b+=2){var c=a[b],e=a[b+1];null!==e.parentNode&&e.parentNode.removeChild(e);var f=c.parentNode;if(f){var g=c.previousSibling,h=0;do{if(c&&8===c.nodeType){var d=c.data;if("/$"===d||"/&"===d)if(0===h)break;else h--;else"$"!==d&&"$?"!==d&&"$~"!==d&&"$!"!==d&&"&"!==d||h++}d=c.nextSibling;f.removeChild(c);c=d}while(c);for(;e.firstChild;)f.insertBefore(e.firstChild,c);g.data="$";g._reactRetry&&requestAnimationFrame(g._reactRetry)}}a.length=0};
$RC=function(a,b){if(b=document.getElementById(b))(a=document.getElementById(a))?(a.previousSibling.data="$~",$RB.push(a,b),2===$RB.length&&("number"!==typeof $RT?requestAnimationFrame($RV.bind(null,$RB)):(a=performance.now(),setTimeout($RV.bind(null,$RB),2300>a&&2E3<a?2300-a:$RT+300-a)))):b.parentNode.removeChild(b)};`)),t.push(`$RC("`)),r=i.toString(16),t.push(e.boundaryPrefix),t.push(r),t.push(`","`),t.push(e.segmentPrefix),t.push(r),a?(t.push(`",`),at(t,n)):t.push(`"`),n=t.push(`)<\/script>`),ze(t,e)&&n}function Or(e,t,n,r){if(r.status===2)return!0;var i=n.contentState,o=r.id;if(o===-1){if((r.id=n.rootSegmentID)===-1)throw Error(a(392));return Er(e,t,r,i)}return o===n.rootSegmentID?Er(e,t,r,i):(Er(e,t,r,i),n=e.resumableState,e=e.renderState,t.push(e.startInlineScript),t.push(`>`),n.instructions&1?t.push(`$RS("`):(n.instructions|=1,t.push(`$RS=function(a,b){a=document.getElementById(a);b=document.getElementById(b);for(a.parentNode.removeChild(a);a.firstChild;)b.parentNode.insertBefore(a.firstChild,b);b.parentNode.removeChild(b)};$RS("`)),t.push(e.segmentPrefix),o=o.toString(16),t.push(o),t.push(`","`),t.push(e.placeholderPrefix),t.push(o),t=t.push(`")<\/script>`),t)}var kr=!1;function Ar(e,t){try{if(!(0<e.pendingRootTasks)){var n,r=e.completedRootSegment;if(r!==null){if(r.status===5)return;var i=e.completedPreambleSegments;if(i===null)return;wr=e.byteSize;var a=e.resumableState,o=e.renderState,s=o.preamble,c=s.htmlChunks,l=s.headChunks,u;if(c){for(u=0;u<c.length;u++)t.push(c[u]);if(l)for(u=0;u<l.length;u++)t.push(l[u]);else{var d=Pe(`head`);t.push(d),t.push(`>`)}}else if(l)for(u=0;u<l.length;u++)t.push(l[u]);var f=o.charsetChunks;for(u=0;u<f.length;u++)t.push(f[u]);f.length=0,o.preconnects.forEach(Qe,t),o.preconnects.clear();var p=o.viewportChunks;for(u=0;u<p.length;u++)t.push(p[u]);p.length=0,o.fontPreloads.forEach(Qe,t),o.fontPreloads.clear(),o.highImagePreloads.forEach(Qe,t),o.highImagePreloads.clear(),oe=o,o.styles.forEach(tt,t),oe=null;var m=o.importMapChunks;for(u=0;u<m.length;u++)t.push(m[u]);m.length=0,o.bootstrapScripts.forEach(Qe,t),o.scripts.forEach(Qe,t),o.scripts.clear(),o.bulkPreloads.forEach(Qe,t),o.bulkPreloads.clear(),a.instructions|=32;var h=o.hoistableChunks;for(u=0;u<h.length;u++)t.push(h[u]);for(a=h.length=0;a<i.length;a++){var g=i[a];for(o=0;o<g.length;o++)Tr(e,t,g[o],null)}var _=e.renderState.preamble,v=_.headChunks;if(_.htmlChunks||v){var y=Le(`head`);t.push(y)}var b=_.bodyChunks;if(b)for(i=0;i<b.length;i++)t.push(b[i]);Tr(e,t,r,null),e.completedRootSegment=null;var x=e.renderState;if(e.allPendingTasks!==0||e.clientRenderedBoundaries.length!==0||e.completedBoundaries.length!==0||e.trackedPostpones!==null&&(e.trackedPostpones.rootNodes.length!==0||e.trackedPostpones.rootSlots!==null)){var S=e.resumableState;if(!(S.instructions&64)){if(S.instructions|=64,t.push(x.startInlineScript),!(S.instructions&32)){S.instructions|=32;var C=`_`+S.idPrefix+`R_`;t.push(` id="`);var w=I(C);t.push(w),t.push(`"`)}t.push(`>`),t.push(`requestAnimationFrame(function(){$RT=performance.now()});`),t.push(`<\/script>`)}}ze(t,x)}var T=e.renderState;r=0;var E=T.viewportChunks;for(r=0;r<E.length;r++)t.push(E[r]);E.length=0,T.preconnects.forEach(Qe,t),T.preconnects.clear(),T.fontPreloads.forEach(Qe,t),T.fontPreloads.clear(),T.highImagePreloads.forEach(Qe,t),T.highImagePreloads.clear(),T.styles.forEach(rt,t),T.scripts.forEach(Qe,t),T.scripts.clear(),T.bulkPreloads.forEach(Qe,t),T.bulkPreloads.clear();var D=T.hoistableChunks;for(r=0;r<D.length;r++)t.push(D[r]);D.length=0;var O=e.clientRenderedBoundaries;for(n=0;n<O.length;n++){var k=O[n];T=t;var A=e.resumableState,j=e.renderState,M=k.rootSegmentID,N=k.errorDigest;T.push(j.startInlineScript),T.push(`>`),A.instructions&4?T.push(`$RX("`):(A.instructions|=4,T.push(`$RX=function(b,c,d,e,f){var a=document.getElementById(b);a&&(b=a.previousSibling,b.data="$!",a=a.dataset,c&&(a.dgst=c),d&&(a.msg=d),e&&(a.stck=e),f&&(a.cstck=f),b._reactRetry&&b._reactRetry())};;$RX("`)),T.push(j.boundaryPrefix);var P=M.toString(16);if(T.push(P),T.push(`"`),N){T.push(`,`);var F=We(N||``);T.push(F)}var ee=T.push(`)<\/script>`);if(!ee){e.destination=null,n++,O.splice(0,n);return}}O.splice(0,n);var te=e.completedBoundaries;for(n=0;n<te.length;n++)if(!Dr(e,t,te[n])){e.destination=null,n++,te.splice(0,n);return}te.splice(0,n),kr=!0;var L=e.partialBoundaries;for(n=0;n<L.length;n++){var R=L[n];a:{O=e,k=t,wr=R.byteSize;var ne=R.completedSegments;for(ee=0;ee<ne.length;ee++)if(!Or(O,k,R,ne[ee])){ee++,ne.splice(0,ee);var re=!1;break a}ne.splice(0,ee);var ie=R.row;ie!==null&&ie.together&&R.pendingTasks===1&&(ie.pendingTasks===1?Jn(O,ie,ie.hoistables):ie.pendingTasks--),re=Ze(k,R.contentState,O.renderState)}if(!re){e.destination=null,n++,L.splice(0,n);return}}L.splice(0,n),kr=!1;var z=e.completedBoundaries;for(n=0;n<z.length;n++)if(!Dr(e,t,z[n])){e.destination=null,n++,z.splice(0,n);return}z.splice(0,n)}}finally{kr=!1,e.allPendingTasks===0&&e.clientRenderedBoundaries.length===0&&e.completedBoundaries.length===0&&(e.flushScheduled=!1,n=e.resumableState,n.hasBody&&(L=Le(`body`),t.push(L)),n.hasHtml&&(n=Le(`html`),t.push(n)),e.status=14,t.push(null),e.destination=null)}}function jr(e){if(!1===e.flushScheduled&&e.pingedTasks.length===0&&e.destination!==null){e.flushScheduled=!0;var t=e.destination;t?Ar(e,t):e.flushScheduled=!1}}function Mr(e,t){if(e.status===13)e.status=14,t.destroy(e.fatalError);else if(e.status!==14&&e.destination===null){e.destination=t;try{Ar(e,t)}catch(t){Gn(e,t,{}),Kn(e,t)}}}function Nr(e,t){(e.status===11||e.status===10)&&(e.status=12);try{var n=e.abortableTasks;if(0<n.size){var r=t===void 0?Error(a(432)):typeof t==`object`&&t&&typeof t.then==`function`?Error(a(530)):t;e.fatalError=r,n.forEach(function(t){return pr(t,e,r)}),n.clear()}e.destination!==null&&Ar(e,e.destination)}catch(t){Gn(e,t,{}),Kn(e,t)}}function Pr(e,t,n){if(t===null)n.rootNodes.push(e);else{var r=n.workingMap,i=r.get(t);i===void 0&&(i=[t[1],t[2],[],null],r.set(t,i),Pr(i,t[0],n)),i[2].push(e)}}function Fr(){}function Ir(e,t,n,r){var i=!1,o=null,s=``,c=!1;if(t=le(t?t.identifierPrefix:void 0),e=Fn(e,t,St(t,n),B(0,null,0,null),1/0,Fr,void 0,function(){c=!0},void 0,void 0,void 0),e.flushScheduled=e.destination!==null,yr(e),e.status===10&&(e.status=11),e.trackedPostpones===null&&mr(e,e.pendingRootTasks===0),Nr(e,r),Mr(e,{push:function(e){return e!==null&&(s+=e),!0},destroy:function(e){i=!0,o=e}}),i&&o!==r)throw o;if(!c)throw Error(a(426));return s}e.renderToStaticMarkup=function(e,t){return Ir(e,t,!0,`The server used "renderToStaticMarkup" which does not support Suspense. If you intended to have the server wait for the suspended component please switch to "renderToReadableStream" which supports Suspense on the server`)},e.renderToString=function(e,t){return Ir(e,t,!1,`The server used "renderToString" which does not support Suspense. If you intended for this Suspense boundary to render the fallback content on the server consider throwing an Error somewhere within the Suspense boundary. If you intended to have the server wait for the suspended component please switch to "renderToReadableStream" which supports Suspense on the server`)},e.version=`19.2.4`})),Cl=t((e=>{var t=n(),r=i();function a(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}var o=Symbol.for(`react.transitional.element`),s=Symbol.for(`react.portal`),c=Symbol.for(`react.fragment`),l=Symbol.for(`react.strict_mode`),u=Symbol.for(`react.profiler`),d=Symbol.for(`react.consumer`),f=Symbol.for(`react.context`),p=Symbol.for(`react.forward_ref`),m=Symbol.for(`react.suspense`),h=Symbol.for(`react.suspense_list`),g=Symbol.for(`react.memo`),_=Symbol.for(`react.lazy`),v=Symbol.for(`react.scope`),y=Symbol.for(`react.activity`),b=Symbol.for(`react.legacy_hidden`),x=Symbol.for(`react.memo_cache_sentinel`),S=Symbol.for(`react.view_transition`),C=Symbol.iterator;function w(e){return typeof e!=`object`||!e?null:(e=C&&e[C]||e[`@@iterator`],typeof e==`function`?e:null)}var T=Array.isArray;function E(e,t){var n=e.length&3,r=e.length-n,i=t;for(t=0;t<r;){var a=e.charCodeAt(t)&255|(e.charCodeAt(++t)&255)<<8|(e.charCodeAt(++t)&255)<<16|(e.charCodeAt(++t)&255)<<24;++t,a=3432918353*(a&65535)+((3432918353*(a>>>16)&65535)<<16)&4294967295,a=a<<15|a>>>17,a=461845907*(a&65535)+((461845907*(a>>>16)&65535)<<16)&4294967295,i^=a,i=i<<13|i>>>19,i=5*(i&65535)+((5*(i>>>16)&65535)<<16)&4294967295,i=(i&65535)+27492+(((i>>>16)+58964&65535)<<16)}switch(a=0,n){case 3:a^=(e.charCodeAt(t+2)&255)<<16;case 2:a^=(e.charCodeAt(t+1)&255)<<8;case 1:a^=e.charCodeAt(t)&255,a=3432918353*(a&65535)+((3432918353*(a>>>16)&65535)<<16)&4294967295,a=a<<15|a>>>17,i^=461845907*(a&65535)+((461845907*(a>>>16)&65535)<<16)&4294967295}return i^=e.length,i^=i>>>16,i=2246822507*(i&65535)+((2246822507*(i>>>16)&65535)<<16)&4294967295,i^=i>>>13,i=3266489909*(i&65535)+((3266489909*(i>>>16)&65535)<<16)&4294967295,(i^i>>>16)>>>0}var D=new MessageChannel,O=[];D.port1.onmessage=function(){var e=O.shift();e&&e()};function k(e){O.push(e),D.port2.postMessage(null)}function A(e){setTimeout(function(){throw e})}var j=Promise,M=typeof queueMicrotask==`function`?queueMicrotask:function(e){j.resolve(null).then(e).catch(A)},N=null,P=0;function F(e,t){if(t.byteLength!==0)if(2048<t.byteLength)0<P&&(e.enqueue(new Uint8Array(N.buffer,0,P)),N=new Uint8Array(2048),P=0),e.enqueue(t);else{var n=N.length-P;n<t.byteLength&&(n===0?e.enqueue(N):(N.set(t.subarray(0,n),P),e.enqueue(N),t=t.subarray(n)),N=new Uint8Array(2048),P=0),N.set(t,P),P+=t.byteLength}}function I(e,t){return F(e,t),!0}function ee(e){N&&0<P&&(e.enqueue(new Uint8Array(N.buffer,0,P)),N=null,P=0)}var te=new TextEncoder;function L(e){return te.encode(e)}function R(e){return te.encode(e)}function ne(e){return e.byteLength}function re(e,t){typeof e.error==`function`?e.error(t):e.close()}var ie=Object.assign,z=Object.prototype.hasOwnProperty,ae=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),oe={},se={};function ce(e){return z.call(se,e)?!0:z.call(oe,e)?!1:ae.test(e)?se[e]=!0:(oe[e]=!0,!1)}var le=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `)),B=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),ue=/["'&<>]/;function V(e){if(typeof e==`boolean`||typeof e==`number`||typeof e==`bigint`)return``+e;e=``+e;var t=ue.exec(e);if(t){var n=``,r,i=0;for(r=t.index;r<e.length;r++){switch(e.charCodeAt(r)){case 34:t=`&quot;`;break;case 38:t=`&amp;`;break;case 39:t=`&#x27;`;break;case 60:t=`&lt;`;break;case 62:t=`&gt;`;break;default:continue}i!==r&&(n+=e.slice(i,r)),i=r+1,n+=t}e=i===r?n:n+e.slice(i,r)}return e}var de=/([A-Z])/g,fe=/^ms-/,pe=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function me(e){return pe.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}var he=t.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,ge=r.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,_e={pending:!1,data:null,method:null,action:null},ve=ge.d;ge.d={f:ve.f,r:ve.r,D:pr,C:mr,L:hr,m:gr,X:vr,S:_r,M:yr};var ye=[],be=null;R(`"></template>`);var xe=R(`<script`),Se=R(`<\/script>`),Ce=R(`<script src="`),we=R(`<script type="module" src="`),Te=R(` nonce="`),Ee=R(` integrity="`),H=R(` crossorigin="`),De=R(` async=""><\/script>`),Oe=R(`<style`),ke=/(<\/|<)(s)(cript)/gi;function U(e,t,n,r){return``+t+(n===`s`?`\\u0073`:`\\u0053`)+r}var Ae=R(`<script type="importmap">`),je=R(`<\/script>`);function Me(e,t,n,r,i,a){n=typeof t==`string`?t:t&&t.script;var o=n===void 0?xe:R(`<script nonce="`+V(n)+`"`),s=typeof t==`string`?void 0:t&&t.style,c=s===void 0?Oe:R(`<style nonce="`+V(s)+`"`),l=e.idPrefix,u=[],d=e.bootstrapScriptContent,f=e.bootstrapScripts,p=e.bootstrapModules;if(d!==void 0&&(u.push(o),ar(u,e),u.push(st,L((``+d).replace(ke,U)),Se)),d=[],r!==void 0&&(d.push(Ae),d.push(L((``+JSON.stringify(r)).replace(ke,U))),d.push(je)),r=i?{preconnects:``,fontPreloads:``,highImagePreloads:``,remainingCapacity:2+(typeof a==`number`?a:2e3)}:null,i={placeholderPrefix:R(l+`P:`),segmentPrefix:R(l+`S:`),boundaryPrefix:R(l+`B:`),startInlineScript:o,startInlineStyle:c,preamble:Pe(),externalRuntimeScript:null,bootstrapChunks:u,importMapChunks:d,onHeaders:i,headers:r,resets:{font:{},dns:{},connect:{default:{},anonymous:{},credentials:{}},image:{},style:{}},charsetChunks:[],viewportChunks:[],hoistableChunks:[],preconnects:new Set,fontPreloads:new Set,highImagePreloads:new Set,styles:new Map,bootstrapScripts:new Set,scripts:new Set,bulkPreloads:new Set,preloads:{images:new Map,stylesheets:new Map,scripts:new Map,moduleScripts:new Map},nonce:{script:n,style:s},hoistableState:null,stylesToHoist:!1},f!==void 0)for(r=0;r<f.length;r++)l=f[r],s=o=void 0,c={rel:`preload`,as:`script`,fetchPriority:`low`,nonce:t},typeof l==`string`?c.href=a=l:(c.href=a=l.src,c.integrity=s=typeof l.integrity==`string`?l.integrity:void 0,c.crossOrigin=o=typeof l==`string`||l.crossOrigin==null?void 0:l.crossOrigin===`use-credentials`?`use-credentials`:``),l=e,d=a,l.scriptResources[d]=null,l.moduleScriptResources[d]=null,l=[],gt(l,c),i.bootstrapScripts.add(l),u.push(Ce,L(V(a)),Xe),n&&u.push(Te,L(V(n)),Xe),typeof s==`string`&&u.push(Ee,L(V(s)),Xe),typeof o==`string`&&u.push(H,L(V(o)),Xe),ar(u,e),u.push(De);if(p!==void 0)for(t=0;t<p.length;t++)s=p[t],a=r=void 0,o={rel:`modulepreload`,fetchPriority:`low`,nonce:n},typeof s==`string`?o.href=f=s:(o.href=f=s.src,o.integrity=a=typeof s.integrity==`string`?s.integrity:void 0,o.crossOrigin=r=typeof s==`string`||s.crossOrigin==null?void 0:s.crossOrigin===`use-credentials`?`use-credentials`:``),s=e,c=f,s.scriptResources[c]=null,s.moduleScriptResources[c]=null,s=[],gt(s,o),i.bootstrapScripts.add(s),u.push(we,L(V(f)),Xe),n&&u.push(Te,L(V(n)),Xe),typeof a==`string`&&u.push(Ee,L(V(a)),Xe),typeof r==`string`&&u.push(H,L(V(r)),Xe),ar(u,e),u.push(De);return i}function Ne(e,t,n,r,i){return{idPrefix:e===void 0?``:e,nextFormID:0,streamingFormat:0,bootstrapScriptContent:n,bootstrapScripts:r,bootstrapModules:i,instructions:0,hasBody:!1,hasHtml:!1,unknownResources:{},dnsResources:{},connectResources:{default:{},anonymous:{},credentials:{}},imageResources:{},styleResources:{},scriptResources:{},moduleUnknownResources:{},moduleScriptResources:{}}}function Pe(){return{htmlChunks:null,headChunks:null,bodyChunks:null}}function Fe(e,t,n,r){return{insertionMode:e,selectedValue:t,tagScope:n,viewTransition:r}}function Ie(e){return Fe(e===`http://www.w3.org/2000/svg`?4:e===`http://www.w3.org/1998/Math/MathML`?5:0,null,0,null)}function Le(e,t,n){var r=e.tagScope&-25;switch(t){case`noscript`:return Fe(2,null,r|1,null);case`select`:return Fe(2,n.value==null?n.defaultValue:n.value,r,null);case`svg`:return Fe(4,null,r,null);case`picture`:return Fe(2,null,r|2,null);case`math`:return Fe(5,null,r,null);case`foreignObject`:return Fe(2,null,r,null);case`table`:return Fe(6,null,r,null);case`thead`:case`tbody`:case`tfoot`:return Fe(7,null,r,null);case`colgroup`:return Fe(9,null,r,null);case`tr`:return Fe(8,null,r,null);case`head`:if(2>e.insertionMode)return Fe(3,null,r,null);break;case`html`:if(e.insertionMode===0)return Fe(1,null,r,null)}return 6<=e.insertionMode||2>e.insertionMode?Fe(2,null,r,null):e.tagScope===r?e:Fe(e.insertionMode,e.selectedValue,r,null)}function Re(e){return e===null?null:{update:e.update,enter:`none`,exit:`none`,share:e.update,name:e.autoName,autoName:e.autoName,nameIdx:0}}function ze(e,t){return t.tagScope&32&&(e.instructions|=128),Fe(t.insertionMode,t.selectedValue,t.tagScope|12,Re(t.viewTransition))}function Be(e,t){e=Re(t.viewTransition);var n=t.tagScope|16;return e!==null&&e.share!==`none`&&(n|=64),Fe(t.insertionMode,t.selectedValue,n,e)}var Ve=R(`<!-- -->`);function He(e,t,n,r){return t===``?r:(r&&e.push(Ve),e.push(L(V(t))),!0)}var Ue=new Map,We=R(` style="`),Ge=R(`:`),Ke=R(`;`);function qe(e,t){if(typeof t!=`object`)throw Error(a(62));var n=!0,r;for(r in t)if(z.call(t,r)){var i=t[r];if(i!=null&&typeof i!=`boolean`&&i!==``){if(r.indexOf(`--`)===0){var o=L(V(r));i=L(V((``+i).trim()))}else o=Ue.get(r),o===void 0&&(o=R(V(r.replace(de,`-$1`).toLowerCase().replace(fe,`-ms-`))),Ue.set(r,o)),i=typeof i==`number`?i===0||le.has(r)?L(``+i):L(i+`px`):L(V((``+i).trim()));n?(n=!1,e.push(We,o,Ge,i)):e.push(Ke,o,Ge,i)}}n||e.push(Xe)}var Je=R(` `),Ye=R(`="`),Xe=R(`"`),Ze=R(`=""`);function Qe(e,t,n){n&&typeof n!=`function`&&typeof n!=`symbol`&&e.push(Je,L(t),Ze)}function $e(e,t,n){typeof n!=`function`&&typeof n!=`symbol`&&typeof n!=`boolean`&&e.push(Je,L(t),Ye,L(V(n)),Xe)}var et=R(V(`javascript:throw new Error('React form unexpectedly submitted.')`)),tt=R(`<input type="hidden"`);function nt(e,t){this.push(tt),rt(e),$e(this,`name`,t),$e(this,`value`,e),this.push(ct)}function rt(e){if(typeof e!=`string`)throw Error(a(480))}function it(e,t){if(typeof t.$$FORM_ACTION==`function`){var n=e.nextFormID++;e=e.idPrefix+n;try{var r=t.$$FORM_ACTION(e);return r&&r.data?.forEach(rt),r}catch(e){if(typeof e==`object`&&e&&typeof e.then==`function`)throw e}}return null}function at(e,t,n,r,i,a,o,s){var c=null;if(typeof r==`function`){var l=it(t,r);l===null?(e.push(Je,L(`formAction`),Ye,et,Xe),o=a=i=r=s=null,pt(t,n)):(s=l.name,r=l.action||``,i=l.encType,a=l.method,o=l.target,c=l.data)}return s!=null&&ot(e,`name`,s),r!=null&&ot(e,`formAction`,r),i!=null&&ot(e,`formEncType`,i),a!=null&&ot(e,`formMethod`,a),o!=null&&ot(e,`formTarget`,o),c}function ot(e,t,n){switch(t){case`className`:$e(e,`class`,n);break;case`tabIndex`:$e(e,`tabindex`,n);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:$e(e,t,n);break;case`style`:qe(e,n);break;case`src`:case`href`:if(n===``)break;case`action`:case`formAction`:if(n==null||typeof n==`function`||typeof n==`symbol`||typeof n==`boolean`)break;n=me(``+n),e.push(Je,L(t),Ye,L(V(n)),Xe);break;case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`ref`:break;case`autoFocus`:case`multiple`:case`muted`:Qe(e,t.toLowerCase(),n);break;case`xlinkHref`:if(typeof n==`function`||typeof n==`symbol`||typeof n==`boolean`)break;n=me(``+n),e.push(Je,L(`xlink:href`),Ye,L(V(n)),Xe);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:typeof n!=`function`&&typeof n!=`symbol`&&e.push(Je,L(t),Ye,L(V(n)),Xe);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:n&&typeof n!=`function`&&typeof n!=`symbol`&&e.push(Je,L(t),Ze);break;case`capture`:case`download`:!0===n?e.push(Je,L(t),Ze):!1!==n&&typeof n!=`function`&&typeof n!=`symbol`&&e.push(Je,L(t),Ye,L(V(n)),Xe);break;case`cols`:case`rows`:case`size`:case`span`:typeof n!=`function`&&typeof n!=`symbol`&&!isNaN(n)&&1<=n&&e.push(Je,L(t),Ye,L(V(n)),Xe);break;case`rowSpan`:case`start`:typeof n==`function`||typeof n==`symbol`||isNaN(n)||e.push(Je,L(t),Ye,L(V(n)),Xe);break;case`xlinkActuate`:$e(e,`xlink:actuate`,n);break;case`xlinkArcrole`:$e(e,`xlink:arcrole`,n);break;case`xlinkRole`:$e(e,`xlink:role`,n);break;case`xlinkShow`:$e(e,`xlink:show`,n);break;case`xlinkTitle`:$e(e,`xlink:title`,n);break;case`xlinkType`:$e(e,`xlink:type`,n);break;case`xmlBase`:$e(e,`xml:base`,n);break;case`xmlLang`:$e(e,`xml:lang`,n);break;case`xmlSpace`:$e(e,`xml:space`,n);break;default:if((!(2<t.length)||t[0]!==`o`&&t[0]!==`O`||t[1]!==`n`&&t[1]!==`N`)&&(t=B.get(t)||t,ce(t))){switch(typeof n){case`function`:case`symbol`:return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`)return}e.push(Je,L(t),Ye,L(V(n)),Xe)}}}var st=R(`>`),ct=R(`/>`);function lt(e,t,n){if(t!=null){if(n!=null)throw Error(a(60));if(typeof t!=`object`||!(`__html`in t))throw Error(a(61));t=t.__html,t!=null&&e.push(L(``+t))}}function ut(e){var n=``;return t.Children.forEach(e,function(e){e!=null&&(n+=e)}),n}var dt=R(` selected=""`),ft=R(`addEventListener("submit",function(a){if(!a.defaultPrevented){var c=a.target,d=a.submitter,e=c.action,b=d;if(d){var f=d.getAttribute("formAction");null!=f&&(e=f,b=null)}"javascript:throw new Error('React form unexpectedly submitted.')"===e&&(a.preventDefault(),b?(a=document.createElement("input"),a.name=b.name,a.value=b.value,b.parentNode.insertBefore(a,b),b=new FormData(c),a.parentNode.removeChild(a)):b=new FormData(c),a=c.ownerDocument||c,(a.$$reactFormReplay=a.$$reactFormReplay||[]).push(c,d,b))}});`);function pt(e,t){if(!(e.instructions&16)){e.instructions|=16;var n=t.preamble,r=t.bootstrapChunks;(n.htmlChunks||n.headChunks)&&r.length===0?(r.push(t.startInlineScript),ar(r,e),r.push(st,ft,Se)):r.unshift(t.startInlineScript,st,ft,Se)}}var mt=R(`<!--F!-->`),ht=R(`<!--F-->`);function gt(e,t){for(var n in e.push(K(`link`)),t)if(z.call(t,n)){var r=t[n];if(r!=null)switch(n){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,`link`));default:ot(e,n,r)}}return e.push(ct),null}var _t=/(<\/|<)(s)(tyle)/gi;function vt(e,t,n,r){return``+t+(n===`s`?`\\73 `:`\\53 `)+r}function yt(e,t,n){for(var r in e.push(K(n)),t)if(z.call(t,r)){var i=t[r];if(i!=null)switch(r){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,n));default:ot(e,r,i)}}return e.push(ct),null}function W(e,t){e.push(K(`title`));var n=null,r=null,i;for(i in t)if(z.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:ot(e,i,a)}}return e.push(st),t=Array.isArray(n)?2>n.length?n[0]:null:n,typeof t!=`function`&&typeof t!=`symbol`&&t!=null&&e.push(L(V(``+t))),lt(e,r,n),e.push(jt(`title`)),null}var G=R(`<!--head-->`),bt=R(`<!--body-->`),xt=R(`<!--html-->`);function St(e,t){e.push(K(`script`));var n=null,r=null,i;for(i in t)if(z.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:ot(e,i,a)}}return e.push(st),lt(e,r,n),typeof n==`string`&&e.push(L((``+n).replace(ke,U))),e.push(jt(`script`)),null}function Ct(e,t,n){e.push(K(n));var r=n=null,i;for(i in t)if(z.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:ot(e,i,a)}}return e.push(st),lt(e,r,n),n}function wt(e,t,n){e.push(K(n));var r=n=null,i;for(i in t)if(z.call(t,i)){var a=t[i];if(a!=null)switch(i){case`children`:n=a;break;case`dangerouslySetInnerHTML`:r=a;break;default:ot(e,i,a)}}return e.push(st),lt(e,r,n),typeof n==`string`?(e.push(L(V(n))),null):n}var Tt=R(`
`),Et=/^[a-zA-Z][a-zA-Z:_\.\-\d]*$/,Dt=new Map;function K(e){var t=Dt.get(e);if(t===void 0){if(!Et.test(e))throw Error(a(65,e));t=R(`<`+e),Dt.set(e,t)}return t}var Ot=R(`<!DOCTYPE html>`);function kt(e,t,n,r,i,o,s,c,l){switch(t){case`div`:case`span`:case`svg`:case`path`:break;case`a`:e.push(K(`a`));var u=null,d=null,f;for(f in n)if(z.call(n,f)){var p=n[f];if(p!=null)switch(f){case`children`:u=p;break;case`dangerouslySetInnerHTML`:d=p;break;case`href`:p===``?$e(e,`href`,``):ot(e,f,p);break;default:ot(e,f,p)}}if(e.push(st),lt(e,d,u),typeof u==`string`){e.push(L(V(u)));var m=null}else m=u;return m;case`g`:case`p`:case`li`:break;case`select`:e.push(K(`select`));var h=null,g=null,_;for(_ in n)if(z.call(n,_)){var v=n[_];if(v!=null)switch(_){case`children`:h=v;break;case`dangerouslySetInnerHTML`:g=v;break;case`defaultValue`:case`value`:break;default:ot(e,_,v)}}return e.push(st),lt(e,g,h),h;case`option`:var y=c.selectedValue;e.push(K(`option`));var b=null,x=null,S=null,C=null,w;for(w in n)if(z.call(n,w)){var E=n[w];if(E!=null)switch(w){case`children`:b=E;break;case`selected`:S=E;break;case`dangerouslySetInnerHTML`:C=E;break;case`value`:x=E;default:ot(e,w,E)}}if(y!=null){var D=x===null?ut(b):``+x;if(T(y)){for(var O=0;O<y.length;O++)if(``+y[O]===D){e.push(dt);break}}else ``+y===D&&e.push(dt)}else S&&e.push(dt);return e.push(st),lt(e,C,b),b;case`textarea`:e.push(K(`textarea`));var k=null,A=null,j=null,M;for(M in n)if(z.call(n,M)){var N=n[M];if(N!=null)switch(M){case`children`:j=N;break;case`value`:k=N;break;case`defaultValue`:A=N;break;case`dangerouslySetInnerHTML`:throw Error(a(91));default:ot(e,M,N)}}if(k===null&&A!==null&&(k=A),e.push(st),j!=null){if(k!=null)throw Error(a(92));if(T(j)){if(1<j.length)throw Error(a(93));k=``+j[0]}k=``+j}return typeof k==`string`&&k[0]===`
`&&e.push(Tt),k!==null&&e.push(L(V(``+k))),null;case`input`:e.push(K(`input`));var P=null,F=null,I=null,ee=null,te=null,R=null,ne=null,re=null,ae=null,oe;for(oe in n)if(z.call(n,oe)){var se=n[oe];if(se!=null)switch(oe){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,`input`));case`name`:P=se;break;case`formAction`:F=se;break;case`formEncType`:I=se;break;case`formMethod`:ee=se;break;case`formTarget`:te=se;break;case`defaultChecked`:ae=se;break;case`defaultValue`:ne=se;break;case`checked`:re=se;break;case`value`:R=se;break;default:ot(e,oe,se)}}var le=at(e,r,i,F,I,ee,te,P);return re===null?ae!==null&&Qe(e,`checked`,ae):Qe(e,`checked`,re),R===null?ne!==null&&ot(e,`value`,ne):ot(e,`value`,R),e.push(ct),le?.forEach(nt,e),null;case`button`:e.push(K(`button`));var B=null,ue=null,de=null,fe=null,pe=null,he=null,ge=null,_e;for(_e in n)if(z.call(n,_e)){var ve=n[_e];if(ve!=null)switch(_e){case`children`:B=ve;break;case`dangerouslySetInnerHTML`:ue=ve;break;case`name`:de=ve;break;case`formAction`:fe=ve;break;case`formEncType`:pe=ve;break;case`formMethod`:he=ve;break;case`formTarget`:ge=ve;break;default:ot(e,_e,ve)}}var be=at(e,r,i,fe,pe,he,ge,de);if(e.push(st),be?.forEach(nt,e),lt(e,ue,B),typeof B==`string`){e.push(L(V(B)));var xe=null}else xe=B;return xe;case`form`:e.push(K(`form`));var Se=null,Ce=null,we=null,Te=null,Ee=null,H=null,De;for(De in n)if(z.call(n,De)){var Oe=n[De];if(Oe!=null)switch(De){case`children`:Se=Oe;break;case`dangerouslySetInnerHTML`:Ce=Oe;break;case`action`:we=Oe;break;case`encType`:Te=Oe;break;case`method`:Ee=Oe;break;case`target`:H=Oe;break;default:ot(e,De,Oe)}}var ke=null,U=null;if(typeof we==`function`){var Ae=it(r,we);Ae===null?(e.push(Je,L(`action`),Ye,et,Xe),H=Ee=Te=we=null,pt(r,i)):(we=Ae.action||``,Te=Ae.encType,Ee=Ae.method,H=Ae.target,ke=Ae.data,U=Ae.name)}if(we!=null&&ot(e,`action`,we),Te!=null&&ot(e,`encType`,Te),Ee!=null&&ot(e,`method`,Ee),H!=null&&ot(e,`target`,H),e.push(st),U!==null&&(e.push(tt),$e(e,`name`,U),e.push(ct),ke?.forEach(nt,e)),lt(e,Ce,Se),typeof Se==`string`){e.push(L(V(Se)));var je=null}else je=Se;return je;case`menuitem`:for(var Me in e.push(K(`menuitem`)),n)if(z.call(n,Me)){var Ne=n[Me];if(Ne!=null)switch(Me){case`children`:case`dangerouslySetInnerHTML`:throw Error(a(400));default:ot(e,Me,Ne)}}return e.push(st),null;case`object`:e.push(K(`object`));var Pe=null,Fe=null,Ie;for(Ie in n)if(z.call(n,Ie)){var Le=n[Ie];if(Le!=null)switch(Ie){case`children`:Pe=Le;break;case`dangerouslySetInnerHTML`:Fe=Le;break;case`data`:var Re=me(``+Le);if(Re===``)break;e.push(Je,L(`data`),Ye,L(V(Re)),Xe);break;default:ot(e,Ie,Le)}}if(e.push(st),lt(e,Fe,Pe),typeof Pe==`string`){e.push(L(V(Pe)));var ze=null}else ze=Pe;return ze;case`title`:var Be=c.tagScope&1,He=c.tagScope&4;if(c.insertionMode===4||Be||n.itemProp!=null)var Ue=W(e,n);else He?Ue=null:(W(i.hoistableChunks,n),Ue=void 0);return Ue;case`link`:var We=c.tagScope&1,Ge=c.tagScope&4,Ke=n.rel,Ze=n.href,rt=n.precedence;if(c.insertionMode===4||We||n.itemProp!=null||typeof Ke!=`string`||typeof Ze!=`string`||Ze===``){gt(e,n);var ft=null}else if(n.rel===`stylesheet`)if(typeof rt!=`string`||n.disabled!=null||n.onLoad||n.onError)ft=gt(e,n);else{var mt=i.styles.get(rt),ht=r.styleResources.hasOwnProperty(Ze)?r.styleResources[Ze]:void 0;if(ht!==null){r.styleResources[Ze]=null,mt||(mt={precedence:L(V(rt)),rules:[],hrefs:[],sheets:new Map},i.styles.set(rt,mt));var Et={state:0,props:ie({},n,{"data-precedence":n.precedence,precedence:null})};if(ht){ht.length===2&&br(Et.props,ht);var Dt=i.preloads.stylesheets.get(Ze);Dt&&0<Dt.length?Dt.length=0:Et.state=1}mt.sheets.set(Ze,Et),s&&s.stylesheets.add(Et)}else if(mt){var kt=mt.sheets.get(Ze);kt&&s&&s.stylesheets.add(kt)}l&&e.push(Ve),ft=null}else n.onLoad||n.onError?ft=gt(e,n):(l&&e.push(Ve),ft=Ge?null:gt(i.hoistableChunks,n));return ft;case`script`:var At=c.tagScope&1,Mt=n.async;if(typeof n.src!=`string`||!n.src||!Mt||typeof Mt==`function`||typeof Mt==`symbol`||n.onLoad||n.onError||c.insertionMode===4||At||n.itemProp!=null)var Nt=St(e,n);else{var q=n.src;if(n.type===`module`)var Pt=r.moduleScriptResources,Ft=i.preloads.moduleScripts;else Pt=r.scriptResources,Ft=i.preloads.scripts;var J=Pt.hasOwnProperty(q)?Pt[q]:void 0;if(J!==null){Pt[q]=null;var It=n;if(J){J.length===2&&(It=ie({},n),br(It,J));var Lt=Ft.get(q);Lt&&(Lt.length=0)}var Rt=[];i.scripts.add(Rt),St(Rt,It)}l&&e.push(Ve),Nt=null}return Nt;case`style`:var zt=c.tagScope&1,Y=n.precedence,Bt=n.href,Vt=n.nonce;if(c.insertionMode===4||zt||n.itemProp!=null||typeof Y!=`string`||typeof Bt!=`string`||Bt===``){e.push(K(`style`));var Ht=null,Ut=null,Wt;for(Wt in n)if(z.call(n,Wt)){var Gt=n[Wt];if(Gt!=null)switch(Wt){case`children`:Ht=Gt;break;case`dangerouslySetInnerHTML`:Ut=Gt;break;default:ot(e,Wt,Gt)}}e.push(st);var Kt=Array.isArray(Ht)?2>Ht.length?Ht[0]:null:Ht;typeof Kt!=`function`&&typeof Kt!=`symbol`&&Kt!=null&&e.push(L((``+Kt).replace(_t,vt))),lt(e,Ut,Ht),e.push(jt(`style`));var qt=null}else{var Jt=i.styles.get(Y);if((r.styleResources.hasOwnProperty(Bt)?r.styleResources[Bt]:void 0)!==null){r.styleResources[Bt]=null,Jt||(Jt={precedence:L(V(Y)),rules:[],hrefs:[],sheets:new Map},i.styles.set(Y,Jt));var Yt=i.nonce.style;if(!Yt||Yt===Vt){Jt.hrefs.push(L(V(Bt)));var Xt=Jt.rules,Zt=null,Qt=null,$t;for($t in n)if(z.call(n,$t)){var en=n[$t];if(en!=null)switch($t){case`children`:Zt=en;break;case`dangerouslySetInnerHTML`:Qt=en}}var tn=Array.isArray(Zt)?2>Zt.length?Zt[0]:null:Zt;typeof tn!=`function`&&typeof tn!=`symbol`&&tn!=null&&Xt.push(L((``+tn).replace(_t,vt))),lt(Xt,Qt,Zt)}}Jt&&s&&s.styles.add(Jt),l&&e.push(Ve),qt=void 0}return qt;case`meta`:var nn=c.tagScope&1,rn=c.tagScope&4;if(c.insertionMode===4||nn||n.itemProp!=null)var an=yt(e,n,`meta`);else l&&e.push(Ve),an=rn?null:typeof n.charSet==`string`?yt(i.charsetChunks,n,`meta`):n.name===`viewport`?yt(i.viewportChunks,n,`meta`):yt(i.hoistableChunks,n,`meta`);return an;case`listing`:case`pre`:e.push(K(t));var on=null,sn=null,cn;for(cn in n)if(z.call(n,cn)){var ln=n[cn];if(ln!=null)switch(cn){case`children`:on=ln;break;case`dangerouslySetInnerHTML`:sn=ln;break;default:ot(e,cn,ln)}}if(e.push(st),sn!=null){if(on!=null)throw Error(a(60));if(typeof sn!=`object`||!(`__html`in sn))throw Error(a(61));var un=sn.__html;un!=null&&(typeof un==`string`&&0<un.length&&un[0]===`
`?e.push(Tt,L(un)):e.push(L(``+un)))}return typeof on==`string`&&on[0]===`
`&&e.push(Tt),on;case`img`:var dn=c.tagScope&3,fn=n.src,pn=n.srcSet;if(!(n.loading===`lazy`||!fn&&!pn||typeof fn!=`string`&&fn!=null||typeof pn!=`string`&&pn!=null||n.fetchPriority===`low`||dn)&&(typeof fn!=`string`||fn[4]!==`:`||fn[0]!==`d`&&fn[0]!==`D`||fn[1]!==`a`&&fn[1]!==`A`||fn[2]!==`t`&&fn[2]!==`T`||fn[3]!==`a`&&fn[3]!==`A`)&&(typeof pn!=`string`||pn[4]!==`:`||pn[0]!==`d`&&pn[0]!==`D`||pn[1]!==`a`&&pn[1]!==`A`||pn[2]!==`t`&&pn[2]!==`T`||pn[3]!==`a`&&pn[3]!==`A`)){s!==null&&c.tagScope&64&&(s.suspenseyImages=!0);var mn=typeof n.sizes==`string`?n.sizes:void 0,hn=pn?pn+`
`+(mn||``):fn,gn=i.preloads.images,_n=gn.get(hn);if(_n)(n.fetchPriority===`high`||10>i.highImagePreloads.size)&&(gn.delete(hn),i.highImagePreloads.add(_n));else if(!r.imageResources.hasOwnProperty(hn)){r.imageResources[hn]=ye;var vn=n.crossOrigin,yn=typeof vn==`string`?vn===`use-credentials`?vn:``:void 0,bn=i.headers,xn;bn&&0<bn.remainingCapacity&&typeof n.srcSet!=`string`&&(n.fetchPriority===`high`||500>bn.highImagePreloads.length)&&(xn=xr(fn,`image`,{imageSrcSet:n.srcSet,imageSizes:n.sizes,crossOrigin:yn,integrity:n.integrity,nonce:n.nonce,type:n.type,fetchPriority:n.fetchPriority,referrerPolicy:n.refererPolicy}),0<=(bn.remainingCapacity-=xn.length+2))?(i.resets.image[hn]=ye,bn.highImagePreloads&&(bn.highImagePreloads+=`, `),bn.highImagePreloads+=xn):(_n=[],gt(_n,{rel:`preload`,as:`image`,href:pn?void 0:fn,imageSrcSet:pn,imageSizes:mn,crossOrigin:yn,integrity:n.integrity,type:n.type,fetchPriority:n.fetchPriority,referrerPolicy:n.referrerPolicy}),n.fetchPriority===`high`||10>i.highImagePreloads.size?i.highImagePreloads.add(_n):(i.bulkPreloads.add(_n),gn.set(hn,_n)))}}return yt(e,n,`img`);case`base`:case`area`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`param`:case`source`:case`track`:case`wbr`:return yt(e,n,t);case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:break;case`head`:if(2>c.insertionMode){var Sn=o||i.preamble;if(Sn.headChunks)throw Error(a(545,"`<head>`"));o!==null&&e.push(G),Sn.headChunks=[];var Cn=Ct(Sn.headChunks,n,`head`)}else Cn=wt(e,n,`head`);return Cn;case`body`:if(2>c.insertionMode){var wn=o||i.preamble;if(wn.bodyChunks)throw Error(a(545,"`<body>`"));o!==null&&e.push(bt),wn.bodyChunks=[];var Tn=Ct(wn.bodyChunks,n,`body`)}else Tn=wt(e,n,`body`);return Tn;case`html`:if(c.insertionMode===0){var En=o||i.preamble;if(En.htmlChunks)throw Error(a(545,"`<html>`"));o!==null&&e.push(xt),En.htmlChunks=[Ot];var Dn=Ct(En.htmlChunks,n,`html`)}else Dn=wt(e,n,`html`);return Dn;default:if(t.indexOf(`-`)!==-1){e.push(K(t));var On=null,kn=null,An;for(An in n)if(z.call(n,An)){var jn=n[An];if(jn!=null){var Mn=An;switch(An){case`children`:On=jn;break;case`dangerouslySetInnerHTML`:kn=jn;break;case`style`:qe(e,jn);break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`ref`:break;case`className`:Mn=`class`;default:if(ce(An)&&typeof jn!=`function`&&typeof jn!=`symbol`&&!1!==jn){if(!0===jn)jn=``;else if(typeof jn==`object`)continue;e.push(Je,L(Mn),Ye,L(V(jn)),Xe)}}}}return e.push(st),lt(e,kn,On),On}}return wt(e,n,t)}var At=new Map;function jt(e){var t=At.get(e);return t===void 0&&(t=R(`</`+e+`>`),At.set(e,t)),t}function Mt(e,t){e=e.preamble,e.htmlChunks===null&&t.htmlChunks&&(e.htmlChunks=t.htmlChunks),e.headChunks===null&&t.headChunks&&(e.headChunks=t.headChunks),e.bodyChunks===null&&t.bodyChunks&&(e.bodyChunks=t.bodyChunks)}function Nt(e,t){t=t.bootstrapChunks;for(var n=0;n<t.length-1;n++)F(e,t[n]);return n<t.length?(n=t[n],t.length=0,I(e,n)):!0}var q=R(`requestAnimationFrame(function(){$RT=performance.now()});`),Pt=R(`<template id="`),Ft=R(`"></template>`),J=R(`<!--&-->`),It=R(`<!--/&-->`),Lt=R(`<!--$-->`),Rt=R(`<!--$?--><template id="`),zt=R(`"></template>`),Y=R(`<!--$!-->`),Bt=R(`<!--/$-->`),Vt=R(`<template`),Ht=R(`"`),Ut=R(` data-dgst="`);R(` data-msg="`),R(` data-stck="`),R(` data-cstck="`);var Wt=R(`></template>`);function Gt(e,t,n){if(F(e,Rt),n===null)throw Error(a(395));return F(e,t.boundaryPrefix),F(e,L(n.toString(16))),I(e,zt)}var Kt=R(`<div hidden id="`),qt=R(`">`),Jt=R(`</div>`),Yt=R(`<svg aria-hidden="true" style="display:none" id="`),Xt=R(`">`),Zt=R(`</svg>`),Qt=R(`<math aria-hidden="true" style="display:none" id="`),$t=R(`">`),en=R(`</math>`),tn=R(`<table hidden id="`),nn=R(`">`),rn=R(`</table>`),an=R(`<table hidden><tbody id="`),on=R(`">`),sn=R(`</tbody></table>`),cn=R(`<table hidden><tr id="`),ln=R(`">`),un=R(`</tr></table>`),dn=R(`<table hidden><colgroup id="`),fn=R(`">`),pn=R(`</colgroup></table>`);function mn(e,t,n,r){switch(n.insertionMode){case 0:case 1:case 3:case 2:return F(e,Kt),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,qt);case 4:return F(e,Yt),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,Xt);case 5:return F(e,Qt),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,$t);case 6:return F(e,tn),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,nn);case 7:return F(e,an),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,on);case 8:return F(e,cn),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,ln);case 9:return F(e,dn),F(e,t.segmentPrefix),F(e,L(r.toString(16))),I(e,fn);default:throw Error(a(397))}}function hn(e,t){switch(t.insertionMode){case 0:case 1:case 3:case 2:return I(e,Jt);case 4:return I(e,Zt);case 5:return I(e,en);case 6:return I(e,rn);case 7:return I(e,sn);case 8:return I(e,un);case 9:return I(e,pn);default:throw Error(a(397))}}var gn=R(`$RS=function(a,b){a=document.getElementById(a);b=document.getElementById(b);for(a.parentNode.removeChild(a);a.firstChild;)b.parentNode.insertBefore(a.firstChild,b);b.parentNode.removeChild(b)};$RS("`),_n=R(`$RS("`),vn=R(`","`),yn=R(`")<\/script>`);R(`<template data-rsi="" data-sid="`),R(`" data-pid="`);var bn=R(`$RB=[];$RV=function(a){$RT=performance.now();for(var b=0;b<a.length;b+=2){var c=a[b],e=a[b+1];null!==e.parentNode&&e.parentNode.removeChild(e);var f=c.parentNode;if(f){var g=c.previousSibling,h=0;do{if(c&&8===c.nodeType){var d=c.data;if("/$"===d||"/&"===d)if(0===h)break;else h--;else"$"!==d&&"$?"!==d&&"$~"!==d&&"$!"!==d&&"&"!==d||h++}d=c.nextSibling;f.removeChild(c);c=d}while(c);for(;e.firstChild;)f.insertBefore(e.firstChild,c);g.data="$";g._reactRetry&&requestAnimationFrame(g._reactRetry)}}a.length=0};
$RC=function(a,b){if(b=document.getElementById(b))(a=document.getElementById(a))?(a.previousSibling.data="$~",$RB.push(a,b),2===$RB.length&&("number"!==typeof $RT?requestAnimationFrame($RV.bind(null,$RB)):(a=performance.now(),setTimeout($RV.bind(null,$RB),2300>a&&2E3<a?2300-a:$RT+300-a)))):b.parentNode.removeChild(b)};`);L(`$RV=function(A,g){function k(a,b){var e=a.getAttribute(b);e&&(b=a.style,l.push(a,b.viewTransitionName,b.viewTransitionClass),"auto"!==e&&(b.viewTransitionClass=e),(a=a.getAttribute("vt-name"))||(a="_T_"+K++ +"_"),b.viewTransitionName=a,B=!0)}var B=!1,K=0,l=[];try{var f=document.__reactViewTransition;if(f){f.finished.finally($RV.bind(null,g));return}var m=new Map;for(f=1;f<g.length;f+=2)for(var h=g[f].querySelectorAll("[vt-share]"),d=0;d<h.length;d++){var c=h[d];m.set(c.getAttribute("vt-name"),c)}var u=[];for(h=0;h<g.length;h+=2){var C=g[h],x=C.parentNode;if(x){var v=x.getBoundingClientRect();if(v.left||v.top||v.width||v.height){c=C;for(f=0;c;){if(8===c.nodeType){var r=c.data;if("/$"===r)if(0===f)break;else f--;else"$"!==r&&"$?"!==r&&"$~"!==r&&"$!"!==r||f++}else if(1===c.nodeType){d=c;var D=d.getAttribute("vt-name"),y=m.get(D);k(d,y?"vt-share":"vt-exit");y&&(k(y,"vt-share"),m.set(D,null));var E=d.querySelectorAll("[vt-share]");for(d=0;d<E.length;d++){var F=E[d],G=F.getAttribute("vt-name"),
H=m.get(G);H&&(k(F,"vt-share"),k(H,"vt-share"),m.set(G,null))}}c=c.nextSibling}for(var I=g[h+1],t=I.firstElementChild;t;)null!==m.get(t.getAttribute("vt-name"))&&k(t,"vt-enter"),t=t.nextElementSibling;c=x;do for(var n=c.firstElementChild;n;){var J=n.getAttribute("vt-update");J&&"none"!==J&&!l.includes(n)&&k(n,"vt-update");n=n.nextElementSibling}while((c=c.parentNode)&&1===c.nodeType&&"none"!==c.getAttribute("vt-update"));u.push.apply(u,I.querySelectorAll('img[src]:not([loading="lazy"])'))}}}if(B){var z=
document.__reactViewTransition=document.startViewTransition({update:function(){A(g);for(var a=[document.documentElement.clientHeight,document.fonts.ready],b={},e=0;e<u.length;b={g:b.g},e++)if(b.g=u[e],!b.g.complete){var p=b.g.getBoundingClientRect();0<p.bottom&&0<p.right&&p.top<window.innerHeight&&p.left<window.innerWidth&&(p=new Promise(function(w){return function(q){w.g.addEventListener("load",q);w.g.addEventListener("error",q)}}(b)),a.push(p))}return Promise.race([Promise.all(a),new Promise(function(w){var q=
performance.now();setTimeout(w,2300>q&&2E3<q?2300-q:500)})])},types:[]});z.ready.finally(function(){for(var a=l.length-3;0<=a;a-=3){var b=l[a],e=b.style;e.viewTransitionName=l[a+1];e.viewTransitionClass=l[a+1];""===b.getAttribute("style")&&b.removeAttribute("style")}});z.finished.finally(function(){document.__reactViewTransition===z&&(document.__reactViewTransition=null)});$RB=[];return}}catch(a){}A(g)}.bind(null,$RV);`);var xn=R(`$RC("`),Sn=R(`$RM=new Map;$RR=function(n,w,p){function u(q){this._p=null;q()}for(var r=new Map,t=document,h,b,e=t.querySelectorAll("link[data-precedence],style[data-precedence]"),v=[],k=0;b=e[k++];)"not all"===b.getAttribute("media")?v.push(b):("LINK"===b.tagName&&$RM.set(b.getAttribute("href"),b),r.set(b.dataset.precedence,h=b));e=0;b=[];var l,a;for(k=!0;;){if(k){var f=p[e++];if(!f){k=!1;e=0;continue}var c=!1,m=0;var d=f[m++];if(a=$RM.get(d)){var g=a._p;c=!0}else{a=t.createElement("link");a.href=d;a.rel=
"stylesheet";for(a.dataset.precedence=l=f[m++];g=f[m++];)a.setAttribute(g,f[m++]);g=a._p=new Promise(function(q,x){a.onload=u.bind(a,q);a.onerror=u.bind(a,x)});$RM.set(d,a)}d=a.getAttribute("media");!g||d&&!matchMedia(d).matches||b.push(g);if(c)continue}else{a=v[e++];if(!a)break;l=a.getAttribute("data-precedence");a.removeAttribute("media")}c=r.get(l)||h;c===h&&(h=a);r.set(l,a);c?c.parentNode.insertBefore(a,c.nextSibling):(c=t.head,c.insertBefore(a,c.firstChild))}if(p=document.getElementById(n))p.previousSibling.data=
"$~";Promise.all(b).then($RC.bind(null,n,w),$RX.bind(null,n,"CSS failed to load"))};$RR("`),Cn=R(`$RR("`),wn=R(`","`),Tn=R(`",`),En=R(`"`),Dn=R(`)<\/script>`);R(`<template data-rci="" data-bid="`),R(`<template data-rri="" data-bid="`),R(`" data-sid="`),R(`" data-sty="`);var On=R(`$RX=function(b,c,d,e,f){var a=document.getElementById(b);a&&(b=a.previousSibling,b.data="$!",a=a.dataset,c&&(a.dgst=c),d&&(a.msg=d),e&&(a.stck=e),f&&(a.cstck=f),b._reactRetry&&b._reactRetry())};`),kn=R(`$RX=function(b,c,d,e,f){var a=document.getElementById(b);a&&(b=a.previousSibling,b.data="$!",a=a.dataset,c&&(a.dgst=c),d&&(a.msg=d),e&&(a.stck=e),f&&(a.cstck=f),b._reactRetry&&b._reactRetry())};;$RX("`),An=R(`$RX("`),jn=R(`"`),Mn=R(`,`),Nn=R(`)<\/script>`);R(`<template data-rxi="" data-bid="`),R(`" data-dgst="`),R(`" data-msg="`),R(`" data-stck="`),R(`" data-cstck="`);var Pn=/[<\u2028\u2029]/g;function Fn(e){return JSON.stringify(e).replace(Pn,function(e){switch(e){case`<`:return`\\u003c`;case`\u2028`:return`\\u2028`;case`\u2029`:return`\\u2029`;default:throw Error(`escapeJSStringsForInstructionScripts encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}})}var In=/[&><\u2028\u2029]/g;function Ln(e){return JSON.stringify(e).replace(In,function(e){switch(e){case`&`:return`\\u0026`;case`>`:return`\\u003e`;case`<`:return`\\u003c`;case`\u2028`:return`\\u2028`;case`\u2029`:return`\\u2029`;default:throw Error(`escapeJSObjectForInstructionScripts encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}})}var Rn=R(` media="not all" data-precedence="`),zn=R(`" data-href="`),Bn=R(`">`),Vn=R(`</style>`),Hn=!1,Un=!0;function Wn(e){var t=e.rules,n=e.hrefs,r=0;if(n.length){for(F(this,be.startInlineStyle),F(this,Rn),F(this,e.precedence),F(this,zn);r<n.length-1;r++)F(this,n[r]),F(this,Qn);for(F(this,n[r]),F(this,Bn),r=0;r<t.length;r++)F(this,t[r]);Un=I(this,Vn),Hn=!0,t.length=0,n.length=0}}function Gn(e){return e.state===2?!1:Hn=!0}function Kn(e,t,n){return Hn=!1,Un=!0,be=n,t.styles.forEach(Wn,e),be=null,t.stylesheets.forEach(Gn),Hn&&(n.stylesToHoist=!0),Un}function qn(e){for(var t=0;t<e.length;t++)F(this,e[t]);e.length=0}var Jn=[];function Yn(e){gt(Jn,e.props);for(var t=0;t<Jn.length;t++)F(this,Jn[t]);Jn.length=0,e.state=2}var Xn=R(` data-precedence="`),Zn=R(`" data-href="`),Qn=R(` `),$n=R(`">`),er=R(`</style>`);function tr(e){var t=0<e.sheets.size;e.sheets.forEach(Yn,this),e.sheets.clear();var n=e.rules,r=e.hrefs;if(!t||r.length){if(F(this,be.startInlineStyle),F(this,Xn),F(this,e.precedence),e=0,r.length){for(F(this,Zn);e<r.length-1;e++)F(this,r[e]),F(this,Qn);F(this,r[e])}for(F(this,$n),e=0;e<n.length;e++)F(this,n[e]);F(this,er),n.length=0,r.length=0}}function nr(e){if(e.state===0){e.state=1;var t=e.props;for(gt(Jn,{rel:`preload`,as:`style`,href:e.props.href,crossOrigin:t.crossOrigin,fetchPriority:t.fetchPriority,integrity:t.integrity,media:t.media,hrefLang:t.hrefLang,referrerPolicy:t.referrerPolicy}),e=0;e<Jn.length;e++)F(this,Jn[e]);Jn.length=0}}function rr(e){e.sheets.forEach(nr,this),e.sheets.clear()}R(`<link rel="expect" href="#`),R(`" blocking="render"/>`);var ir=R(` id="`);function ar(e,t){!(t.instructions&32)&&(t.instructions|=32,e.push(ir,L(V(`_`+t.idPrefix+`R_`)),Xe))}var or=R(`[`),sr=R(`,[`),cr=R(`,`),lr=R(`]`);function ur(e,t){F(e,or);var n=or;t.stylesheets.forEach(function(t){if(t.state!==2)if(t.state===3)F(e,n),F(e,L(Ln(``+t.props.href))),F(e,lr),n=sr;else{F(e,n);var r=t.props[`data-precedence`],i=t.props;for(var o in F(e,L(Ln(me(``+t.props.href)))),r=``+r,F(e,cr),F(e,L(Ln(r))),i)if(z.call(i,o)&&(r=i[o],r!=null))switch(o){case`href`:case`rel`:case`precedence`:case`data-precedence`:break;case`children`:case`dangerouslySetInnerHTML`:throw Error(a(399,`link`));default:dr(e,o,r)}F(e,lr),n=sr,t.state=3}}),F(e,lr)}function dr(e,t,n){var r=t.toLowerCase();switch(typeof n){case`function`:case`symbol`:return}switch(t){case`innerHTML`:case`dangerouslySetInnerHTML`:case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`style`:case`ref`:return;case`className`:r=`class`,t=``+n;break;case`hidden`:if(!1===n)return;t=``;break;case`src`:case`href`:n=me(n),t=``+n;break;default:if(2<t.length&&(t[0]===`o`||t[0]===`O`)&&(t[1]===`n`||t[1]===`N`)||!ce(t))return;t=``+n}F(e,cr),F(e,L(Ln(r))),F(e,cr),F(e,L(Ln(t)))}function fr(){return{styles:new Set,stylesheets:new Set,suspenseyImages:!1}}function pr(e){var t=qi||null;if(t){var n=t.resumableState,r=t.renderState;if(typeof e==`string`&&e){if(!n.dnsResources.hasOwnProperty(e)){n.dnsResources[e]=null,n=r.headers;var i,a;(a=n&&0<n.remainingCapacity)&&(a=(i=`<`+(``+e).replace(Sr,Cr)+`>; rel=dns-prefetch`,0<=(n.remainingCapacity-=i.length+2))),a?(r.resets.dns[e]=null,n.preconnects&&(n.preconnects+=`, `),n.preconnects+=i):(i=[],gt(i,{href:e,rel:`dns-prefetch`}),r.preconnects.add(i))}Wa(t)}}else ve.D(e)}function mr(e,t){var n=qi||null;if(n){var r=n.resumableState,i=n.renderState;if(typeof e==`string`&&e){var a=t===`use-credentials`?`credentials`:typeof t==`string`?`anonymous`:`default`;if(!r.connectResources[a].hasOwnProperty(e)){r.connectResources[a][e]=null,r=i.headers;var o,s;if(s=r&&0<r.remainingCapacity){if(s=`<`+(``+e).replace(Sr,Cr)+`>; rel=preconnect`,typeof t==`string`){var c=(``+t).replace(wr,Tr);s+=`; crossorigin="`+c+`"`}s=(o=s,0<=(r.remainingCapacity-=o.length+2))}s?(i.resets.connect[a][e]=null,r.preconnects&&(r.preconnects+=`, `),r.preconnects+=o):(a=[],gt(a,{rel:`preconnect`,href:e,crossOrigin:t}),i.preconnects.add(a))}Wa(n)}}else ve.C(e,t)}function hr(e,t,n){var r=qi||null;if(r){var i=r.resumableState,a=r.renderState;if(t&&e){switch(t){case`image`:if(n)var o=n.imageSrcSet,s=n.imageSizes,c=n.fetchPriority;var l=o?o+`
`+(s||``):e;if(i.imageResources.hasOwnProperty(l))return;i.imageResources[l]=ye,i=a.headers;var u;i&&0<i.remainingCapacity&&typeof o!=`string`&&c===`high`&&(u=xr(e,t,n),0<=(i.remainingCapacity-=u.length+2))?(a.resets.image[l]=ye,i.highImagePreloads&&(i.highImagePreloads+=`, `),i.highImagePreloads+=u):(i=[],gt(i,ie({rel:`preload`,href:o?void 0:e,as:t},n)),c===`high`?a.highImagePreloads.add(i):(a.bulkPreloads.add(i),a.preloads.images.set(l,i)));break;case`style`:if(i.styleResources.hasOwnProperty(e))return;o=[],gt(o,ie({rel:`preload`,href:e,as:t},n)),i.styleResources[e]=!n||typeof n.crossOrigin!=`string`&&typeof n.integrity!=`string`?ye:[n.crossOrigin,n.integrity],a.preloads.stylesheets.set(e,o),a.bulkPreloads.add(o);break;case`script`:if(i.scriptResources.hasOwnProperty(e))return;o=[],a.preloads.scripts.set(e,o),a.bulkPreloads.add(o),gt(o,ie({rel:`preload`,href:e,as:t},n)),i.scriptResources[e]=!n||typeof n.crossOrigin!=`string`&&typeof n.integrity!=`string`?ye:[n.crossOrigin,n.integrity];break;default:if(i.unknownResources.hasOwnProperty(t)){if(o=i.unknownResources[t],o.hasOwnProperty(e))return}else o={},i.unknownResources[t]=o;if(o[e]=ye,(i=a.headers)&&0<i.remainingCapacity&&t===`font`&&(l=xr(e,t,n),0<=(i.remainingCapacity-=l.length+2)))a.resets.font[e]=ye,i.fontPreloads&&(i.fontPreloads+=`, `),i.fontPreloads+=l;else switch(i=[],e=ie({rel:`preload`,href:e,as:t},n),gt(i,e),t){case`font`:a.fontPreloads.add(i);break;default:a.bulkPreloads.add(i)}}Wa(r)}}else ve.L(e,t,n)}function gr(e,t){var n=qi||null;if(n){var r=n.resumableState,i=n.renderState;if(e){var a=t&&typeof t.as==`string`?t.as:`script`;switch(a){case`script`:if(r.moduleScriptResources.hasOwnProperty(e))return;a=[],r.moduleScriptResources[e]=!t||typeof t.crossOrigin!=`string`&&typeof t.integrity!=`string`?ye:[t.crossOrigin,t.integrity],i.preloads.moduleScripts.set(e,a);break;default:if(r.moduleUnknownResources.hasOwnProperty(a)){var o=r.unknownResources[a];if(o.hasOwnProperty(e))return}else o={},r.moduleUnknownResources[a]=o;a=[],o[e]=ye}gt(a,ie({rel:`modulepreload`,href:e},t)),i.bulkPreloads.add(a),Wa(n)}}else ve.m(e,t)}function _r(e,t,n){var r=qi||null;if(r){var i=r.resumableState,a=r.renderState;if(e){t||=`default`;var o=a.styles.get(t),s=i.styleResources.hasOwnProperty(e)?i.styleResources[e]:void 0;s!==null&&(i.styleResources[e]=null,o||(o={precedence:L(V(t)),rules:[],hrefs:[],sheets:new Map},a.styles.set(t,o)),t={state:0,props:ie({rel:`stylesheet`,href:e,"data-precedence":t},n)},s&&(s.length===2&&br(t.props,s),(a=a.preloads.stylesheets.get(e))&&0<a.length?a.length=0:t.state=1),o.sheets.set(e,t),Wa(r))}}else ve.S(e,t,n)}function vr(e,t){var n=qi||null;if(n){var r=n.resumableState,i=n.renderState;if(e){var a=r.scriptResources.hasOwnProperty(e)?r.scriptResources[e]:void 0;a!==null&&(r.scriptResources[e]=null,t=ie({src:e,async:!0},t),a&&(a.length===2&&br(t,a),e=i.preloads.scripts.get(e))&&(e.length=0),e=[],i.scripts.add(e),St(e,t),Wa(n))}}else ve.X(e,t)}function yr(e,t){var n=qi||null;if(n){var r=n.resumableState,i=n.renderState;if(e){var a=r.moduleScriptResources.hasOwnProperty(e)?r.moduleScriptResources[e]:void 0;a!==null&&(r.moduleScriptResources[e]=null,t=ie({src:e,type:`module`,async:!0},t),a&&(a.length===2&&br(t,a),e=i.preloads.moduleScripts.get(e))&&(e.length=0),e=[],i.scripts.add(e),St(e,t),Wa(n))}}else ve.M(e,t)}function br(e,t){e.crossOrigin??=t[0],e.integrity??=t[1]}function xr(e,t,n){for(var r in e=(``+e).replace(Sr,Cr),t=(``+t).replace(wr,Tr),t=`<`+e+`>; rel=preload; as="`+t+`"`,n)z.call(n,r)&&(e=n[r],typeof e==`string`&&(t+=`; `+r.toLowerCase()+`="`+(``+e).replace(wr,Tr)+`"`));return t}var Sr=/[<>\r\n]/g;function Cr(e){switch(e){case`<`:return`%3C`;case`>`:return`%3E`;case`
`:return`%0A`;case`\r`:return`%0D`;default:throw Error(`escapeLinkHrefForHeaderContextReplacer encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}}var wr=/["';,\r\n]/g;function Tr(e){switch(e){case`"`:return`%22`;case`'`:return`%27`;case`;`:return`%3B`;case`,`:return`%2C`;case`
`:return`%0A`;case`\r`:return`%0D`;default:throw Error(`escapeStringForLinkHeaderQuotedParamValueContextReplacer encountered a match it does not know how to replace. this means the match regex and the replacement characters are no longer in sync. This is a bug in React`)}}function Er(e){this.styles.add(e)}function Dr(e){this.stylesheets.add(e)}function Or(e,t){t.styles.forEach(Er,e),t.stylesheets.forEach(Dr,e),t.suspenseyImages&&(e.suspenseyImages=!0)}function kr(e){return 0<e.stylesheets.size||e.suspenseyImages}var Ar=Function.prototype.bind,jr=Symbol.for(`react.client.reference`);function Mr(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===jr?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case c:return`Fragment`;case u:return`Profiler`;case l:return`StrictMode`;case m:return`Suspense`;case h:return`SuspenseList`;case y:return`Activity`}if(typeof e==`object`)switch(e.$$typeof){case s:return`Portal`;case f:return e.displayName||`Context`;case d:return(e._context.displayName||`Context`)+`.Consumer`;case p:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case g:return t=e.displayName||null,t===null?Mr(e.type)||`Memo`:t;case _:t=e._payload,e=e._init;try{return Mr(e(t))}catch{}}return null}var Nr={},Pr=null;function Fr(e,t){if(e!==t){e.context._currentValue=e.parentValue,e=e.parent;var n=t.parent;if(e===null){if(n!==null)throw Error(a(401))}else{if(n===null)throw Error(a(401));Fr(e,n)}t.context._currentValue=t.value}}function Ir(e){e.context._currentValue=e.parentValue,e=e.parent,e!==null&&Ir(e)}function Lr(e){var t=e.parent;t!==null&&Lr(t),e.context._currentValue=e.value}function Rr(e,t){if(e.context._currentValue=e.parentValue,e=e.parent,e===null)throw Error(a(402));e.depth===t.depth?Fr(e,t):Rr(e,t)}function zr(e,t){var n=t.parent;if(n===null)throw Error(a(402));e.depth===n.depth?Fr(e,n):zr(e,n),t.context._currentValue=t.value}function Br(e){var t=Pr;t!==e&&(t===null?Lr(e):e===null?Ir(t):t.depth===e.depth?Fr(t,e):t.depth>e.depth?Rr(t,e):zr(t,e),Pr=e)}var Vr={enqueueSetState:function(e,t){e=e._reactInternals,e.queue!==null&&e.queue.push(t)},enqueueReplaceState:function(e,t){e=e._reactInternals,e.replace=!0,e.queue=[t]},enqueueForceUpdate:function(){}},Hr={id:1,overflow:``};function Ur(e,t,n){var r=e.id;e=e.overflow;var i=32-Wr(r)-1;r&=~(1<<i),n+=1;var a=32-Wr(t)+i;if(30<a){var o=i-i%5;return a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,{id:1<<32-Wr(t)+i|n<<i|r,overflow:a+e}}return{id:1<<a|n<<i|r,overflow:e}}var Wr=Math.clz32?Math.clz32:qr,Gr=Math.log,Kr=Math.LN2;function qr(e){return e>>>=0,e===0?32:31-(Gr(e)/Kr|0)|0}function Jr(){}var Yr=Error(a(460));function Xr(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(Jr,Jr),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw t.reason;default:switch(typeof t.status==`string`?t.then(Jr,Jr):(e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})),t.status){case`fulfilled`:return t.value;case`rejected`:throw t.reason}throw Zr=t,Yr}}var Zr=null;function Qr(){if(Zr===null)throw Error(a(459));var e=Zr;return Zr=null,e}function $r(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var ei=typeof Object.is==`function`?Object.is:$r,ti=null,ni=null,ri=null,ii=null,ai=null,oi=null,si=!1,ci=!1,li=0,ui=0,di=-1,fi=0,pi=null,mi=null,hi=0;function gi(){if(ti===null)throw Error(a(321));return ti}function _i(){if(0<hi)throw Error(a(312));return{memoizedState:null,queue:null,next:null}}function vi(){return oi===null?ai===null?(si=!1,ai=oi=_i()):(si=!0,oi=ai):oi.next===null?(si=!1,oi=oi.next=_i()):(si=!0,oi=oi.next),oi}function yi(){var e=pi;return pi=null,e}function bi(){ii=ri=ni=ti=null,ci=!1,ai=null,hi=0,oi=mi=null}function xi(e,t){return typeof t==`function`?t(e):t}function Si(e,t,n){if(ti=gi(),oi=vi(),si){var r=oi.queue;if(t=r.dispatch,mi!==null&&(n=mi.get(r),n!==void 0)){mi.delete(r),r=oi.memoizedState;do r=e(r,n.action),n=n.next;while(n!==null);return oi.memoizedState=r,[r,t]}return[oi.memoizedState,t]}return e=e===xi?typeof t==`function`?t():t:n===void 0?t:n(t),oi.memoizedState=e,e=oi.queue={last:null,dispatch:null},e=e.dispatch=wi.bind(null,ti,e),[oi.memoizedState,e]}function Ci(e,t){if(ti=gi(),oi=vi(),t=t===void 0?null:t,oi!==null){var n=oi.memoizedState;if(n!==null&&t!==null){var r=n[1];a:if(r===null)r=!1;else{for(var i=0;i<r.length&&i<t.length;i++)if(!ei(t[i],r[i])){r=!1;break a}r=!0}if(r)return n[0]}}return e=e(),oi.memoizedState=[e,t],e}function wi(e,t,n){if(25<=hi)throw Error(a(301));if(e===ti)if(ci=!0,e={action:n,next:null},mi===null&&(mi=new Map),n=mi.get(t),n===void 0)mi.set(t,e);else{for(t=n;t.next!==null;)t=t.next;t.next=e}}function Ti(){throw Error(a(440))}function Ei(){throw Error(a(394))}function Di(){throw Error(a(479))}function Oi(e,t,n){gi();var r=ui++,i=ri;if(typeof e.$$FORM_ACTION==`function`){var a=null,o=ii;i=i.formState;var s=e.$$IS_SIGNATURE_EQUAL;if(i!==null&&typeof s==`function`){var c=i[1];s.call(e,i[2],i[3])&&(a=n===void 0?`k`+E(JSON.stringify([o,null,r]),0):`p`+n,c===a&&(di=r,t=i[0]))}var l=e.bind(null,t);return e=function(e){l(e)},typeof l.$$FORM_ACTION==`function`&&(e.$$FORM_ACTION=function(e){e=l.$$FORM_ACTION(e),n!==void 0&&(n+=``,e.action=n);var t=e.data;return t&&(a===null&&(a=n===void 0?`k`+E(JSON.stringify([o,null,r]),0):`p`+n),t.append(`$ACTION_KEY`,a)),e}),[t,e,!1]}var u=e.bind(null,t);return[t,function(e){u(e)},!1]}function ki(e){var t=fi;return fi+=1,pi===null&&(pi=[]),Xr(pi,e,t)}function Ai(){throw Error(a(393))}var ji={readContext:function(e){return e._currentValue},use:function(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return ki(e);if(e.$$typeof===f)return e._currentValue}throw Error(a(438,String(e)))},useContext:function(e){return gi(),e._currentValue},useMemo:Ci,useReducer:Si,useRef:function(e){ti=gi(),oi=vi();var t=oi.memoizedState;return t===null?(e={current:e},oi.memoizedState=e):t},useState:function(e){return Si(xi,e)},useInsertionEffect:Jr,useLayoutEffect:Jr,useCallback:function(e,t){return Ci(function(){return e},t)},useImperativeHandle:Jr,useEffect:Jr,useDebugValue:Jr,useDeferredValue:function(e,t){return gi(),t===void 0?e:t},useTransition:function(){return gi(),[!1,Ei]},useId:function(){var e=ni.treeContext,t=e.overflow;e=e.id,e=(e&~(1<<32-Wr(e)-1)).toString(32)+t;var n=Mi;if(n===null)throw Error(a(404));return t=li++,e=`_`+n.idPrefix+`R_`+e,0<t&&(e+=`H`+t.toString(32)),e+`_`},useSyncExternalStore:function(e,t,n){if(n===void 0)throw Error(a(407));return n()},useOptimistic:function(e){return gi(),[e,Di]},useActionState:Oi,useFormState:Oi,useHostTransitionStatus:function(){return gi(),_e},useMemoCache:function(e){for(var t=Array(e),n=0;n<e;n++)t[n]=x;return t},useCacheRefresh:function(){return Ai},useEffectEvent:function(){return Ti}},Mi=null,Ni={getCacheForType:function(){throw Error(a(248))},cacheSignal:function(){throw Error(a(248))}},Pi,Fi;function Ii(e){if(Pi===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);Pi=t&&t[1]||``,Fi=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+Pi+e+Fi}var Li=!1;function Ri(e,t){if(!e||Li)return``;Li=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,`props`,{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}e.call(n.prototype)}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,`name`,{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{Li=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?Ii(n):``}function zi(e){if(typeof e==`string`)return Ii(e);if(typeof e==`function`)return e.prototype&&e.prototype.isReactComponent?Ri(e,!0):Ri(e,!1);if(typeof e==`object`&&e){switch(e.$$typeof){case p:return Ri(e.render,!1);case g:return Ri(e.type,!1);case _:var t=e,n=t._payload;t=t._init;try{e=t(n)}catch{return Ii(`Lazy`)}return zi(e)}if(typeof e.name==`string`){a:{n=e.name,t=e.env;var r=e.debugLocation;if(r!=null&&(e=Error.prepareStackTrace,Error.prepareStackTrace=void 0,r=r.stack,Error.prepareStackTrace=e,r.startsWith(`Error: react-stack-top-frame
`)&&(r=r.slice(29)),e=r.indexOf(`
`),e!==-1&&(r=r.slice(e+1)),e=r.indexOf(`react_stack_bottom_frame`),e!==-1&&(e=r.lastIndexOf(`
`,e)),e=e===-1?``:r=r.slice(0,e),r=e.lastIndexOf(`
`),e=r===-1?e:e.slice(r+1),e.indexOf(n)!==-1)){n=`
`+e;break a}n=Ii(n+(t?` [`+t+`]`:``))}return n}}switch(e){case h:return Ii(`SuspenseList`);case m:return Ii(`Suspense`)}return``}function Bi(e,t){return(500<t.byteSize||kr(t.contentState))&&t.contentPreamble===null}function Vi(e){if(typeof e==`object`&&e&&typeof e.environmentName==`string`){var t=e.environmentName;e=[e].slice(0),typeof e[0]==`string`?e.splice(0,1,`%c%s%c `+e[0],`background: #e6e6e6;background: light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.25));color: #000000;color: light-dark(#000000, #ffffff);border-radius: 2px`,` `+t+` `,``):e.splice(0,0,`%c%s%c`,`background: #e6e6e6;background: light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.25));color: #000000;color: light-dark(#000000, #ffffff);border-radius: 2px`,` `+t+` `,``),e.unshift(console),t=Ar.apply(console.error,e),t()}else console.error(e);return null}function Hi(e,t,n,r,i,a,o,s,c,l,u){var d=new Set;this.destination=null,this.flushScheduled=!1,this.resumableState=e,this.renderState=t,this.rootFormatContext=n,this.progressiveChunkSize=r===void 0?12800:r,this.status=10,this.fatalError=null,this.pendingRootTasks=this.allPendingTasks=this.nextSegmentId=0,this.completedPreambleSegments=this.completedRootSegment=null,this.byteSize=0,this.abortableTasks=d,this.pingedTasks=[],this.clientRenderedBoundaries=[],this.completedBoundaries=[],this.partialBoundaries=[],this.trackedPostpones=null,this.onError=i===void 0?Vi:i,this.onPostpone=l===void 0?Jr:l,this.onAllReady=a===void 0?Jr:a,this.onShellReady=o===void 0?Jr:o,this.onShellError=s===void 0?Jr:s,this.onFatalError=c===void 0?Jr:c,this.formState=u===void 0?null:u}function Ui(e,t,n,r,i,a,o,s,c,l,u,d){return t=new Hi(t,n,r,i,a,o,s,c,l,u,d),n=Qi(t,0,null,r,!1,!1),n.parentFlushed=!0,e=Xi(t,null,e,-1,null,n,null,null,t.abortableTasks,null,r,null,Hr,null,null),$i(e),t.pingedTasks.push(e),t}function Wi(e,t,n,r,i,a,o,s,c,l,u){return e=Ui(e,t,n,r,i,a,o,s,c,l,u,void 0),e.trackedPostpones={workingMap:new Map,rootNodes:[],rootSlots:null},e}function Gi(e,t,n,r,i,a,o,s,c){return n=new Hi(t.resumableState,n,t.rootFormatContext,t.progressiveChunkSize,r,i,a,o,s,c,null),n.nextSegmentId=t.nextSegmentId,typeof t.replaySlots==`number`?(r=Qi(n,0,null,t.rootFormatContext,!1,!1),r.parentFlushed=!0,e=Xi(n,null,e,-1,null,r,null,null,n.abortableTasks,null,t.rootFormatContext,null,Hr,null,null),$i(e),n.pingedTasks.push(e),n):(e=Zi(n,null,{nodes:t.replayNodes,slots:t.replaySlots,pendingTasks:0},e,-1,null,null,n.abortableTasks,null,t.rootFormatContext,null,Hr,null,null),$i(e),n.pingedTasks.push(e),n)}function Ki(e,t,n,r,i,a,o,s,c){return e=Gi(e,t,n,r,i,a,o,s,c),e.trackedPostpones={workingMap:new Map,rootNodes:[],rootSlots:null},e}var qi=null;function Ji(e,t){e.pingedTasks.push(t),e.pingedTasks.length===1&&(e.flushScheduled=e.destination!==null,e.trackedPostpones!==null||e.status===10?M(function(){return ja(e)}):k(function(){return ja(e)}))}function Yi(e,t,n,r,i){return n={status:0,rootSegmentID:-1,parentFlushed:!1,pendingTasks:0,row:t,completedSegments:[],byteSize:0,fallbackAbortableTasks:n,errorDigest:null,contentState:fr(),fallbackState:fr(),contentPreamble:r,fallbackPreamble:i,trackedContentKeyPath:null,trackedFallbackNode:null},t!==null&&(t.pendingTasks++,r=t.boundaries,r!==null&&(e.allPendingTasks++,n.pendingTasks++,r.push(n)),e=t.inheritedHoistables,e!==null&&Or(n.contentState,e)),n}function Xi(e,t,n,r,i,a,o,s,c,l,u,d,f,p,m){e.allPendingTasks++,i===null?e.pendingRootTasks++:i.pendingTasks++,p!==null&&p.pendingTasks++;var h={replay:null,node:n,childIndex:r,ping:function(){return Ji(e,h)},blockedBoundary:i,blockedSegment:a,blockedPreamble:o,hoistableState:s,abortSet:c,keyPath:l,formatContext:u,context:d,treeContext:f,row:p,componentStack:m,thenableState:t};return c.add(h),h}function Zi(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.allPendingTasks++,a===null?e.pendingRootTasks++:a.pendingTasks++,f!==null&&f.pendingTasks++,n.pendingTasks++;var m={replay:n,node:r,childIndex:i,ping:function(){return Ji(e,m)},blockedBoundary:a,blockedSegment:null,blockedPreamble:null,hoistableState:o,abortSet:s,keyPath:c,formatContext:l,context:u,treeContext:d,row:f,componentStack:p,thenableState:t};return s.add(m),m}function Qi(e,t,n,r,i,a){return{status:0,parentFlushed:!1,id:-1,index:t,chunks:[],children:[],preambleChildren:[],parentFormatContext:r,boundary:n,lastPushedText:i,textEmbedded:a}}function $i(e){var t=e.node;if(typeof t==`object`&&t)switch(t.$$typeof){case o:e.componentStack={parent:e.componentStack,type:t.type}}}function ea(e){return e===null?null:{parent:e.parent,type:`Suspense Fallback`}}function ta(e){var t={};return e&&Object.defineProperty(t,`componentStack`,{configurable:!0,enumerable:!0,get:function(){try{var n=``,r=e;do n+=zi(r.type),r=r.parent;while(r);var i=n}catch(e){i=`
Error generating stack: `+e.message+`
`+e.stack}return Object.defineProperty(t,`componentStack`,{value:i}),i}}),t}function na(e,t,n){if(e=e.onError,t=e(t,n),t==null||typeof t==`string`)return t}function ra(e,t){var n=e.onShellError,r=e.onFatalError;n(t),r(t),e.destination===null?(e.status=13,e.fatalError=t):(e.status=14,re(e.destination,t))}function ia(e,t){aa(e,t.next,t.hoistables)}function aa(e,t,n){for(;t!==null;){n!==null&&(Or(t.hoistables,n),t.inheritedHoistables=n);var r=t.boundaries;if(r!==null){t.boundaries=null;for(var i=0;i<r.length;i++){var a=r[i];n!==null&&Or(a.contentState,n),Aa(e,a,null,null)}}if(t.pendingTasks--,0<t.pendingTasks)break;n=t.hoistables,t=t.next}}function oa(e,t){var n=t.boundaries;if(n!==null&&t.pendingTasks===n.length){for(var r=!0,i=0;i<n.length;i++){var a=n[i];if(a.pendingTasks!==1||a.parentFlushed||Bi(e,a)){r=!1;break}}r&&aa(e,t,t.hoistables)}}function sa(e){var t={pendingTasks:1,boundaries:null,hoistables:fr(),inheritedHoistables:null,together:!1,next:null};return e!==null&&0<e.pendingTasks&&(t.pendingTasks++,t.boundaries=[],e.next=t),t}function ca(e,t,n,r,i){var a=t.keyPath,o=t.treeContext,s=t.row;t.keyPath=n,n=r.length;var c=null;if(t.replay!==null){var l=t.replay.slots;if(typeof l==`object`&&l)for(var u=0;u<n;u++){var d=i!==`backwards`&&i!==`unstable_legacy-backwards`?u:n-1-u,f=r[d];t.row=c=sa(c),t.treeContext=Ur(o,n,d);var p=l[d];typeof p==`number`?(fa(e,t,p,f,d),delete l[d]):xa(e,t,f,d),--c.pendingTasks===0&&ia(e,c)}else for(l=0;l<n;l++)u=i!==`backwards`&&i!==`unstable_legacy-backwards`?l:n-1-l,d=r[u],t.row=c=sa(c),t.treeContext=Ur(o,n,u),xa(e,t,d,u),--c.pendingTasks===0&&ia(e,c)}else if(i!==`backwards`&&i!==`unstable_legacy-backwards`)for(i=0;i<n;i++)l=r[i],t.row=c=sa(c),t.treeContext=Ur(o,n,i),xa(e,t,l,i),--c.pendingTasks===0&&ia(e,c);else{for(i=t.blockedSegment,l=i.children.length,u=i.chunks.length,d=n-1;0<=d;d--){f=r[d],t.row=c=sa(c),t.treeContext=Ur(o,n,d),p=Qi(e,u,null,t.formatContext,d===0?i.lastPushedText:!0,!0),i.children.splice(l,0,p),t.blockedSegment=p;try{xa(e,t,f,d),p.lastPushedText&&p.textEmbedded&&p.chunks.push(Ve),p.status=1,ka(e,t.blockedBoundary,p),--c.pendingTasks===0&&ia(e,c)}catch(t){throw p.status=e.status===12?3:4,t}}t.blockedSegment=i,i.lastPushedText=!1}s!==null&&c!==null&&0<c.pendingTasks&&(s.pendingTasks++,c.next=s),t.treeContext=o,t.row=s,t.keyPath=a}function la(e,t,n,r,i,a){var o=t.thenableState;for(t.thenableState=null,ti={},ni=t,ri=e,ii=n,ui=li=0,di=-1,fi=0,pi=o,e=r(i,a);ci;)ci=!1,ui=li=0,di=-1,fi=0,hi+=1,oi=null,e=r(i,a);return bi(),e}function ua(e,t,n,r,i,a,o){var s=!1;if(a!==0&&e.formState!==null){var c=t.blockedSegment;if(c!==null){s=!0,c=c.chunks;for(var l=0;l<a;l++)l===o?c.push(mt):c.push(ht)}}a=t.keyPath,t.keyPath=n,i?(n=t.treeContext,t.treeContext=Ur(n,1,0),xa(e,t,r,-1),t.treeContext=n):s?xa(e,t,r,-1):pa(e,t,r,-1),t.keyPath=a}function da(e,t,n,r,i,o){if(typeof r==`function`)if(r.prototype&&r.prototype.isReactComponent){var s=i;if(`ref`in i)for(var x in s={},i)x!==`ref`&&(s[x]=i[x]);var C=r.defaultProps;if(C)for(var E in s===i&&(s=ie({},s,i)),C)s[E]===void 0&&(s[E]=C[E]);i=s,s=Nr,C=r.contextType,typeof C==`object`&&C&&(s=C._currentValue),s=new r(i,s);var D=s.state===void 0?null:s.state;if(s.updater=Vr,s.props=i,s.state=D,C={queue:[],replace:!1},s._reactInternals=C,o=r.contextType,s.context=typeof o==`object`&&o?o._currentValue:Nr,o=r.getDerivedStateFromProps,typeof o==`function`&&(o=o(i,D),D=o==null?D:ie({},D,o),s.state=D),typeof r.getDerivedStateFromProps!=`function`&&typeof s.getSnapshotBeforeUpdate!=`function`&&(typeof s.UNSAFE_componentWillMount==`function`||typeof s.componentWillMount==`function`))if(r=s.state,typeof s.componentWillMount==`function`&&s.componentWillMount(),typeof s.UNSAFE_componentWillMount==`function`&&s.UNSAFE_componentWillMount(),r!==s.state&&Vr.enqueueReplaceState(s,s.state,null),C.queue!==null&&0<C.queue.length)if(r=C.queue,o=C.replace,C.queue=null,C.replace=!1,o&&r.length===1)s.state=r[0];else{for(C=o?r[0]:s.state,D=!0,o=o?1:0;o<r.length;o++)E=r[o],E=typeof E==`function`?E.call(s,C,i,void 0):E,E!=null&&(D?(D=!1,C=ie({},C,E)):ie(C,E));s.state=C}else C.queue=null;if(r=s.render(),e.status===12)throw null;i=t.keyPath,t.keyPath=n,pa(e,t,r,-1),t.keyPath=i}else{if(r=la(e,t,n,r,i,void 0),e.status===12)throw null;ua(e,t,n,r,li!==0,ui,di)}else if(typeof r==`string`)if(s=t.blockedSegment,s===null)s=i.children,C=t.formatContext,D=t.keyPath,t.formatContext=Le(C,r,i),t.keyPath=n,xa(e,t,s,-1),t.formatContext=C,t.keyPath=D;else{if(D=kt(s.chunks,r,i,e.resumableState,e.renderState,t.blockedPreamble,t.hoistableState,t.formatContext,s.lastPushedText),s.lastPushedText=!1,C=t.formatContext,o=t.keyPath,t.keyPath=n,(t.formatContext=Le(C,r,i)).insertionMode===3){n=Qi(e,0,null,t.formatContext,!1,!1),s.preambleChildren.push(n),t.blockedSegment=n;try{n.status=6,xa(e,t,D,-1),n.lastPushedText&&n.textEmbedded&&n.chunks.push(Ve),n.status=1,ka(e,t.blockedBoundary,n)}finally{t.blockedSegment=s}}else xa(e,t,D,-1);t.formatContext=C,t.keyPath=o;a:{switch(t=s.chunks,e=e.resumableState,r){case`title`:case`style`:case`script`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`img`:case`input`:case`keygen`:case`link`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:break a;case`body`:if(1>=C.insertionMode){e.hasBody=!0;break a}break;case`html`:if(C.insertionMode===0){e.hasHtml=!0;break a}break;case`head`:if(1>=C.insertionMode)break a}t.push(jt(r))}s.lastPushedText=!1}else{switch(r){case b:case l:case u:case c:r=t.keyPath,t.keyPath=n,pa(e,t,i.children,-1),t.keyPath=r;return;case y:r=t.blockedSegment,r===null?i.mode!==`hidden`&&(r=t.keyPath,t.keyPath=n,xa(e,t,i.children,-1),t.keyPath=r):i.mode!==`hidden`&&(r.chunks.push(J),r.lastPushedText=!1,s=t.keyPath,t.keyPath=n,xa(e,t,i.children,-1),t.keyPath=s,r.chunks.push(It),r.lastPushedText=!1);return;case h:a:{if(r=i.children,i=i.revealOrder,i===`forwards`||i===`backwards`||i===`unstable_legacy-backwards`){if(T(r)){ca(e,t,n,r,i);break a}if((s=w(r))&&(s=s.call(r))){if(C=s.next(),!C.done){do C=s.next();while(!C.done);ca(e,t,n,r,i)}break a}}i===`together`?(i=t.keyPath,s=t.row,C=t.row=sa(null),C.boundaries=[],C.together=!0,t.keyPath=n,pa(e,t,r,-1),--C.pendingTasks===0&&ia(e,C),t.keyPath=i,t.row=s,s!==null&&0<C.pendingTasks&&(s.pendingTasks++,C.next=s)):(i=t.keyPath,t.keyPath=n,pa(e,t,r,-1),t.keyPath=i)}return;case S:case v:throw Error(a(343));case m:a:if(t.replay!==null){r=t.keyPath,s=t.formatContext,C=t.row,t.keyPath=n,t.formatContext=Be(e.resumableState,s),t.row=null,n=i.children;try{xa(e,t,n,-1)}finally{t.keyPath=r,t.formatContext=s,t.row=C}}else{r=t.keyPath,o=t.formatContext;var O=t.row;E=t.blockedBoundary,x=t.blockedPreamble;var k=t.hoistableState,A=t.blockedSegment,j=i.fallback;i=i.children;var M=new Set,N=2>t.formatContext.insertionMode?Yi(e,t.row,M,Pe(),Pe()):Yi(e,t.row,M,null,null);e.trackedPostpones!==null&&(N.trackedContentKeyPath=n);var P=Qi(e,A.chunks.length,N,t.formatContext,!1,!1);A.children.push(P),A.lastPushedText=!1;var F=Qi(e,0,null,t.formatContext,!1,!1);if(F.parentFlushed=!0,e.trackedPostpones!==null){s=t.componentStack,C=[n[0],`Suspense Fallback`,n[2]],D=[C[1],C[2],[],null],e.trackedPostpones.workingMap.set(C,D),N.trackedFallbackNode=D,t.blockedSegment=P,t.blockedPreamble=N.fallbackPreamble,t.keyPath=C,t.formatContext=ze(e.resumableState,o),t.componentStack=ea(s),P.status=6;try{xa(e,t,j,-1),P.lastPushedText&&P.textEmbedded&&P.chunks.push(Ve),P.status=1,ka(e,E,P)}catch(t){throw P.status=e.status===12?3:4,t}finally{t.blockedSegment=A,t.blockedPreamble=x,t.keyPath=r,t.formatContext=o}t=Xi(e,null,i,-1,N,F,N.contentPreamble,N.contentState,t.abortSet,n,Be(e.resumableState,t.formatContext),t.context,t.treeContext,null,s),$i(t),e.pingedTasks.push(t)}else{t.blockedBoundary=N,t.blockedPreamble=N.contentPreamble,t.hoistableState=N.contentState,t.blockedSegment=F,t.keyPath=n,t.formatContext=Be(e.resumableState,o),t.row=null,F.status=6;try{if(xa(e,t,i,-1),F.lastPushedText&&F.textEmbedded&&F.chunks.push(Ve),F.status=1,ka(e,N,F),Oa(N,F),N.pendingTasks===0&&N.status===0){if(N.status=1,!Bi(e,N)){O!==null&&--O.pendingTasks===0&&ia(e,O),e.pendingRootTasks===0&&t.blockedPreamble&&Pa(e);break a}}else O!==null&&O.together&&oa(e,O)}catch(n){N.status=4,e.status===12?(F.status=3,s=e.fatalError):(F.status=4,s=n),C=ta(t.componentStack),D=na(e,s,C),N.errorDigest=D,va(e,N)}finally{t.blockedBoundary=E,t.blockedPreamble=x,t.hoistableState=k,t.blockedSegment=A,t.keyPath=r,t.formatContext=o,t.row=O}t=Xi(e,null,j,-1,E,P,N.fallbackPreamble,N.fallbackState,M,[n[0],`Suspense Fallback`,n[2]],ze(e.resumableState,t.formatContext),t.context,t.treeContext,t.row,ea(t.componentStack)),$i(t),e.pingedTasks.push(t)}}return}if(typeof r==`object`&&r)switch(r.$$typeof){case p:if(`ref`in i)for(A in s={},i)A!==`ref`&&(s[A]=i[A]);else s=i;r=la(e,t,n,r.render,s,o),ua(e,t,n,r,li!==0,ui,di);return;case g:da(e,t,n,r.type,i,o);return;case f:if(C=i.children,s=t.keyPath,i=i.value,D=r._currentValue,r._currentValue=i,o=Pr,Pr=r={parent:o,depth:o===null?0:o.depth+1,context:r,parentValue:D,value:i},t.context=r,t.keyPath=n,pa(e,t,C,-1),e=Pr,e===null)throw Error(a(403));e.context._currentValue=e.parentValue,e=Pr=e.parent,t.context=e,t.keyPath=s;return;case d:i=i.children,r=i(r._context._currentValue),i=t.keyPath,t.keyPath=n,pa(e,t,r,-1),t.keyPath=i;return;case _:if(s=r._init,r=s(r._payload),e.status===12)throw null;da(e,t,n,r,i,o);return}throw Error(a(130,r==null?r:typeof r,``))}}function fa(e,t,n,r,i){var a=t.replay,o=t.blockedBoundary,s=Qi(e,0,null,t.formatContext,!1,!1);s.id=n,s.parentFlushed=!0;try{t.replay=null,t.blockedSegment=s,xa(e,t,r,i),s.status=1,ka(e,o,s),o===null?e.completedRootSegment=s:(Oa(o,s),o.parentFlushed&&e.partialBoundaries.push(o))}finally{t.replay=a,t.blockedSegment=null}}function pa(e,t,n,r){t.replay!==null&&typeof t.replay.slots==`number`?fa(e,t,t.replay.slots,n,r):(t.node=n,t.childIndex=r,n=t.componentStack,$i(t),ma(e,t),t.componentStack=n)}function ma(e,t){var n=t.node,r=t.childIndex;if(n!==null){if(typeof n==`object`){switch(n.$$typeof){case o:var i=n.type,c=n.key,l=n.props;n=l.ref;var u=n===void 0?null:n,d=Mr(i),p=c??(r===-1?0:r);if(c=[t.keyPath,d,p],t.replay!==null)a:{var h=t.replay;for(r=h.nodes,n=0;n<r.length;n++){var g=r[n];if(p===g[1]){if(g.length===4){if(d!==null&&d!==g[0])throw Error(a(490,g[0],d));var v=g[2];d=g[3],p=t.node,t.replay={nodes:v,slots:d,pendingTasks:1};try{if(da(e,t,c,i,l,u),t.replay.pendingTasks===1&&0<t.replay.nodes.length)throw Error(a(488));t.replay.pendingTasks--}catch(a){if(typeof a==`object`&&a&&(a===Yr||typeof a.then==`function`))throw t.node===p?t.replay=h:r.splice(n,1),a;t.replay.pendingTasks--,l=ta(t.componentStack),c=e,e=t.blockedBoundary,i=a,l=na(c,i,l),Ca(c,e,v,d,i,l)}t.replay=h}else{if(i!==m)throw Error(a(490,`Suspense`,Mr(i)||`Unknown`));b:{h=void 0,i=g[5],u=g[2],d=g[3],p=g[4]===null?[]:g[4][2],g=g[4]===null?null:g[4][3];var y=t.keyPath,b=t.formatContext,x=t.row,S=t.replay,C=t.blockedBoundary,E=t.hoistableState,D=l.children,O=l.fallback,k=new Set;l=2>t.formatContext.insertionMode?Yi(e,t.row,k,Pe(),Pe()):Yi(e,t.row,k,null,null),l.parentFlushed=!0,l.rootSegmentID=i,t.blockedBoundary=l,t.hoistableState=l.contentState,t.keyPath=c,t.formatContext=Be(e.resumableState,b),t.row=null,t.replay={nodes:u,slots:d,pendingTasks:1};try{if(xa(e,t,D,-1),t.replay.pendingTasks===1&&0<t.replay.nodes.length)throw Error(a(488));if(t.replay.pendingTasks--,l.pendingTasks===0&&l.status===0){l.status=1,e.completedBoundaries.push(l);break b}}catch(n){l.status=4,v=ta(t.componentStack),h=na(e,n,v),l.errorDigest=h,t.replay.pendingTasks--,e.clientRenderedBoundaries.push(l)}finally{t.blockedBoundary=C,t.hoistableState=E,t.replay=S,t.keyPath=y,t.formatContext=b,t.row=x}v=Zi(e,null,{nodes:p,slots:g,pendingTasks:0},O,-1,C,l.fallbackState,k,[c[0],`Suspense Fallback`,c[2]],ze(e.resumableState,t.formatContext),t.context,t.treeContext,t.row,ea(t.componentStack)),$i(v),e.pingedTasks.push(v)}}r.splice(n,1);break a}}}else da(e,t,c,i,l,u);return;case s:throw Error(a(257));case _:if(v=n._init,n=v(n._payload),e.status===12)throw null;pa(e,t,n,r);return}if(T(n)){ha(e,t,n,r);return}if((v=w(n))&&(v=v.call(n))){if(n=v.next(),!n.done){l=[];do l.push(n.value),n=v.next();while(!n.done);ha(e,t,l,r)}return}if(typeof n.then==`function`)return t.thenableState=null,pa(e,t,ki(n),r);if(n.$$typeof===f)return pa(e,t,n._currentValue,r);throw r=Object.prototype.toString.call(n),Error(a(31,r===`[object Object]`?`object with keys {`+Object.keys(n).join(`, `)+`}`:r))}typeof n==`string`?(r=t.blockedSegment,r!==null&&(r.lastPushedText=He(r.chunks,n,e.renderState,r.lastPushedText))):(typeof n==`number`||typeof n==`bigint`)&&(r=t.blockedSegment,r!==null&&(r.lastPushedText=He(r.chunks,``+n,e.renderState,r.lastPushedText)))}}function ha(e,t,n,r){var i=t.keyPath;if(r!==-1&&(t.keyPath=[t.keyPath,`Fragment`,r],t.replay!==null)){for(var o=t.replay,s=o.nodes,c=0;c<s.length;c++){var l=s[c];if(l[1]===r){r=l[2],l=l[3],t.replay={nodes:r,slots:l,pendingTasks:1};try{if(ha(e,t,n,-1),t.replay.pendingTasks===1&&0<t.replay.nodes.length)throw Error(a(488));t.replay.pendingTasks--}catch(i){if(typeof i==`object`&&i&&(i===Yr||typeof i.then==`function`))throw i;t.replay.pendingTasks--,n=ta(t.componentStack);var u=t.blockedBoundary,d=i;n=na(e,d,n),Ca(e,u,r,l,d,n)}t.replay=o,s.splice(c,1);break}}t.keyPath=i;return}if(o=t.treeContext,s=n.length,t.replay!==null&&(c=t.replay.slots,typeof c==`object`&&c)){for(r=0;r<s;r++)l=n[r],t.treeContext=Ur(o,s,r),u=c[r],typeof u==`number`?(fa(e,t,u,l,r),delete c[r]):xa(e,t,l,r);t.treeContext=o,t.keyPath=i;return}for(c=0;c<s;c++)r=n[c],t.treeContext=Ur(o,s,c),xa(e,t,r,c);t.treeContext=o,t.keyPath=i}function ga(e,t,n){if(n.status=5,n.rootSegmentID=e.nextSegmentId++,e=n.trackedContentKeyPath,e===null)throw Error(a(486));var r=n.trackedFallbackNode,i=[],o=t.workingMap.get(e);return o===void 0?(n=[e[1],e[2],i,null,r,n.rootSegmentID],t.workingMap.set(e,n),qa(n,e[0],t),n):(o[4]=r,o[5]=n.rootSegmentID,o)}function _a(e,t,n,r){r.status=5;var i=n.keyPath,o=n.blockedBoundary;if(o===null)r.id=e.nextSegmentId++,t.rootSlots=r.id,e.completedRootSegment!==null&&(e.completedRootSegment.status=5);else{if(o!==null&&o.status===0){var s=ga(e,t,o);if(o.trackedContentKeyPath===i&&n.childIndex===-1){r.id===-1&&(r.id=r.parentFlushed?o.rootSegmentID:e.nextSegmentId++),s[3]=r.id;return}}if(r.id===-1&&(r.id=r.parentFlushed&&o!==null?o.rootSegmentID:e.nextSegmentId++),n.childIndex===-1)i===null?t.rootSlots=r.id:(n=t.workingMap.get(i),n===void 0?(n=[i[1],i[2],[],r.id],qa(n,i[0],t)):n[3]=r.id);else{if(i===null){if(e=t.rootSlots,e===null)e=t.rootSlots={};else if(typeof e==`number`)throw Error(a(491))}else if(o=t.workingMap,s=o.get(i),s===void 0)e={},s=[i[1],i[2],[],e],o.set(i,s),qa(s,i[0],t);else if(e=s[3],e===null)e=s[3]={};else if(typeof e==`number`)throw Error(a(491));e[n.childIndex]=r.id}}}function va(e,t){e=e.trackedPostpones,e!==null&&(t=t.trackedContentKeyPath,t!==null&&(t=e.workingMap.get(t),t!==void 0&&(t.length=4,t[2]=[],t[3]=null)))}function ya(e,t,n){return Zi(e,n,t.replay,t.node,t.childIndex,t.blockedBoundary,t.hoistableState,t.abortSet,t.keyPath,t.formatContext,t.context,t.treeContext,t.row,t.componentStack)}function ba(e,t,n){var r=t.blockedSegment,i=Qi(e,r.chunks.length,null,t.formatContext,r.lastPushedText,!0);return r.children.push(i),r.lastPushedText=!1,Xi(e,n,t.node,t.childIndex,t.blockedBoundary,i,t.blockedPreamble,t.hoistableState,t.abortSet,t.keyPath,t.formatContext,t.context,t.treeContext,t.row,t.componentStack)}function xa(e,t,n,r){var i=t.formatContext,a=t.context,o=t.keyPath,s=t.treeContext,c=t.componentStack,l=t.blockedSegment;if(l===null){l=t.replay;try{return pa(e,t,n,r)}catch(u){if(bi(),n=u===Yr?Qr():u,e.status!==12&&typeof n==`object`&&n){if(typeof n.then==`function`){r=u===Yr?yi():null,e=ya(e,t,r).ping,n.then(e,e),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,t.replay=l,Br(a);return}if(n.message===`Maximum call stack size exceeded`){n=u===Yr?yi():null,n=ya(e,t,n),e.pingedTasks.push(n),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,t.replay=l,Br(a);return}}}}else{var u=l.children.length,d=l.chunks.length;try{return pa(e,t,n,r)}catch(r){if(bi(),l.children.length=u,l.chunks.length=d,n=r===Yr?Qr():r,e.status!==12&&typeof n==`object`&&n){if(typeof n.then==`function`){l=n,n=r===Yr?yi():null,e=ba(e,t,n).ping,l.then(e,e),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,Br(a);return}if(n.message===`Maximum call stack size exceeded`){l=r===Yr?yi():null,l=ba(e,t,l),e.pingedTasks.push(l),t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,t.componentStack=c,Br(a);return}}}}throw t.formatContext=i,t.context=a,t.keyPath=o,t.treeContext=s,Br(a),n}function Sa(e){var t=e.blockedBoundary,n=e.blockedSegment;n!==null&&(n.status=3,Aa(this,t,e.row,n))}function Ca(e,t,n,r,i,o){for(var s=0;s<n.length;s++){var c=n[s];if(c.length===4)Ca(e,t,c[2],c[3],i,o);else{c=c[5];var l=e,u=o,d=Yi(l,null,new Set,null,null);d.parentFlushed=!0,d.rootSegmentID=c,d.status=4,d.errorDigest=u,d.parentFlushed&&l.clientRenderedBoundaries.push(d)}}if(n.length=0,r!==null){if(t===null)throw Error(a(487));if(t.status!==4&&(t.status=4,t.errorDigest=o,t.parentFlushed&&e.clientRenderedBoundaries.push(t)),typeof r==`object`)for(var f in r)delete r[f]}}function wa(e,t,n){var r=e.blockedBoundary,i=e.blockedSegment;if(i!==null){if(i.status===6)return;i.status=3}var a=ta(e.componentStack);if(r===null){if(t.status!==13&&t.status!==14){if(r=e.replay,r===null){t.trackedPostpones!==null&&i!==null?(r=t.trackedPostpones,na(t,n,a),_a(t,r,e,i),Aa(t,null,e.row,i)):(na(t,n,a),ra(t,n));return}r.pendingTasks--,r.pendingTasks===0&&0<r.nodes.length&&(i=na(t,n,a),Ca(t,null,r.nodes,r.slots,n,i)),t.pendingRootTasks--,t.pendingRootTasks===0&&Ea(t)}}else{var o=t.trackedPostpones;if(r.status!==4){if(o!==null&&i!==null)return na(t,n,a),_a(t,o,e,i),r.fallbackAbortableTasks.forEach(function(e){return wa(e,t,n)}),r.fallbackAbortableTasks.clear(),Aa(t,r,e.row,i);r.status=4,i=na(t,n,a),r.status=4,r.errorDigest=i,va(t,r),r.parentFlushed&&t.clientRenderedBoundaries.push(r)}r.pendingTasks--,i=r.row,i!==null&&--i.pendingTasks===0&&ia(t,i),r.fallbackAbortableTasks.forEach(function(e){return wa(e,t,n)}),r.fallbackAbortableTasks.clear()}e=e.row,e!==null&&--e.pendingTasks===0&&ia(t,e),t.allPendingTasks--,t.allPendingTasks===0&&Da(t)}function Ta(e,t){try{var n=e.renderState,r=n.onHeaders;if(r){var i=n.headers;if(i){n.headers=null;var a=i.preconnects;if(i.fontPreloads&&(a&&(a+=`, `),a+=i.fontPreloads),i.highImagePreloads&&(a&&(a+=`, `),a+=i.highImagePreloads),!t){var o=n.styles.values(),s=o.next();b:for(;0<i.remainingCapacity&&!s.done;s=o.next())for(var c=s.value.sheets.values(),l=c.next();0<i.remainingCapacity&&!l.done;l=c.next()){var u=l.value,d=u.props,f=d.href,p=u.props,m=xr(p.href,`style`,{crossOrigin:p.crossOrigin,integrity:p.integrity,nonce:p.nonce,type:p.type,fetchPriority:p.fetchPriority,referrerPolicy:p.referrerPolicy,media:p.media});if(0<=(i.remainingCapacity-=m.length+2))n.resets.style[f]=ye,a&&(a+=`, `),a+=m,n.resets.style[f]=typeof d.crossOrigin==`string`||typeof d.integrity==`string`?[d.crossOrigin,d.integrity]:ye;else break b}}r(a?{Link:a}:{})}}}catch(t){na(e,t,{})}}function Ea(e){e.trackedPostpones===null&&Ta(e,!0),e.trackedPostpones===null&&Pa(e),e.onShellError=Jr,e=e.onShellReady,e()}function Da(e){Ta(e,e.trackedPostpones===null?!0:e.completedRootSegment===null||e.completedRootSegment.status!==5),Pa(e),e=e.onAllReady,e()}function Oa(e,t){if(t.chunks.length===0&&t.children.length===1&&t.children[0].boundary===null&&t.children[0].id===-1){var n=t.children[0];n.id=t.id,n.parentFlushed=!0,n.status!==1&&n.status!==3&&n.status!==4||Oa(e,n)}else e.completedSegments.push(t)}function ka(e,t,n){if(ne!==null){n=n.chunks;for(var r=0,i=0;i<n.length;i++)r+=n[i].byteLength;t===null?e.byteSize+=r:t.byteSize+=r}}function Aa(e,t,n,r){if(n!==null&&(--n.pendingTasks===0?ia(e,n):n.together&&oa(e,n)),e.allPendingTasks--,t===null){if(r!==null&&r.parentFlushed){if(e.completedRootSegment!==null)throw Error(a(389));e.completedRootSegment=r}e.pendingRootTasks--,e.pendingRootTasks===0&&Ea(e)}else if(t.pendingTasks--,t.status!==4)if(t.pendingTasks===0){if(t.status===0&&(t.status=1),r!==null&&r.parentFlushed&&(r.status===1||r.status===3)&&Oa(t,r),t.parentFlushed&&e.completedBoundaries.push(t),t.status===1)n=t.row,n!==null&&Or(n.hoistables,t.contentState),Bi(e,t)||(t.fallbackAbortableTasks.forEach(Sa,e),t.fallbackAbortableTasks.clear(),n!==null&&--n.pendingTasks===0&&ia(e,n)),e.pendingRootTasks===0&&e.trackedPostpones===null&&t.contentPreamble!==null&&Pa(e);else if(t.status===5&&(t=t.row,t!==null)){if(e.trackedPostpones!==null){n=e.trackedPostpones;var i=t.next;if(i!==null&&(r=i.boundaries,r!==null))for(i.boundaries=null,i=0;i<r.length;i++){var o=r[i];ga(e,n,o),Aa(e,o,null,null)}}--t.pendingTasks===0&&ia(e,t)}}else r===null||!r.parentFlushed||r.status!==1&&r.status!==3||(Oa(t,r),t.completedSegments.length===1&&t.parentFlushed&&e.partialBoundaries.push(t)),t=t.row,t!==null&&t.together&&oa(e,t);e.allPendingTasks===0&&Da(e)}function ja(e){if(e.status!==14&&e.status!==13){var t=Pr,n=he.H;he.H=ji;var r=he.A;he.A=Ni;var i=qi;qi=e;var o=Mi;Mi=e.resumableState;try{var s=e.pingedTasks,c;for(c=0;c<s.length;c++){var l=s[c],u=e,d=l.blockedSegment;if(d===null){var f=u;if(l.replay.pendingTasks!==0){Br(l.context);try{if(typeof l.replay.slots==`number`?fa(f,l,l.replay.slots,l.node,l.childIndex):ma(f,l),l.replay.pendingTasks===1&&0<l.replay.nodes.length)throw Error(a(488));l.replay.pendingTasks--,l.abortSet.delete(l),Aa(f,l.blockedBoundary,l.row,null)}catch(e){bi();var p=e===Yr?Qr():e;if(typeof p==`object`&&p&&typeof p.then==`function`){var m=l.ping;p.then(m,m),l.thenableState=e===Yr?yi():null}else{l.replay.pendingTasks--,l.abortSet.delete(l);var h=ta(l.componentStack);u=void 0;var g=f,_=l.blockedBoundary,v=f.status===12?f.fatalError:p,y=l.replay.nodes,b=l.replay.slots;u=na(g,v,h),Ca(g,_,y,b,v,u),f.pendingRootTasks--,f.pendingRootTasks===0&&Ea(f),f.allPendingTasks--,f.allPendingTasks===0&&Da(f)}}}}else if(f=void 0,g=d,g.status===0){g.status=6,Br(l.context);var x=g.children.length,S=g.chunks.length;try{ma(u,l),g.lastPushedText&&g.textEmbedded&&g.chunks.push(Ve),l.abortSet.delete(l),g.status=1,ka(u,l.blockedBoundary,g),Aa(u,l.blockedBoundary,l.row,g)}catch(e){bi(),g.children.length=x,g.chunks.length=S;var C=e===Yr?Qr():u.status===12?u.fatalError:e;if(u.status===12&&u.trackedPostpones!==null){var w=u.trackedPostpones,T=ta(l.componentStack);l.abortSet.delete(l),na(u,C,T),_a(u,w,l,g),Aa(u,l.blockedBoundary,l.row,g)}else if(typeof C==`object`&&C&&typeof C.then==`function`){g.status=0,l.thenableState=e===Yr?yi():null;var E=l.ping;C.then(E,E)}else{var D=ta(l.componentStack);l.abortSet.delete(l),g.status=4;var O=l.blockedBoundary,k=l.row;if(k!==null&&--k.pendingTasks===0&&ia(u,k),u.allPendingTasks--,f=na(u,C,D),O===null)ra(u,C);else if(O.pendingTasks--,O.status!==4){O.status=4,O.errorDigest=f,va(u,O);var A=O.row;A!==null&&--A.pendingTasks===0&&ia(u,A),O.parentFlushed&&u.clientRenderedBoundaries.push(O),u.pendingRootTasks===0&&u.trackedPostpones===null&&O.contentPreamble!==null&&Pa(u)}u.allPendingTasks===0&&Da(u)}}}}s.splice(0,c),e.destination!==null&&Ha(e,e.destination)}catch(t){na(e,t,{}),ra(e,t)}finally{Mi=o,he.H=n,he.A=r,n===ji&&Br(t),qi=i}}}function Ma(e,t,n){t.preambleChildren.length&&n.push(t.preambleChildren);for(var r=!1,i=0;i<t.children.length;i++)r=Na(e,t.children[i],n)||r;return r}function Na(e,t,n){var r=t.boundary;if(r===null)return Ma(e,t,n);var i=r.contentPreamble,o=r.fallbackPreamble;if(i===null||o===null)return!1;switch(r.status){case 1:if(Mt(e.renderState,i),e.byteSize+=r.byteSize,t=r.completedSegments[0],!t)throw Error(a(391));return Ma(e,t,n);case 5:if(e.trackedPostpones!==null)return!0;case 4:if(t.status===1)return Mt(e.renderState,o),Ma(e,t,n);default:return!0}}function Pa(e){if(e.completedRootSegment&&e.completedPreambleSegments===null){var t=[],n=e.byteSize,r=Na(e,e.completedRootSegment,t),i=e.renderState.preamble;!1===r||i.headChunks&&i.bodyChunks?e.completedPreambleSegments=t:e.byteSize=n}}function Fa(e,t,n,r){switch(n.parentFlushed=!0,n.status){case 0:n.id=e.nextSegmentId++;case 5:return r=n.id,n.lastPushedText=!1,n.textEmbedded=!1,e=e.renderState,F(t,Pt),F(t,e.placeholderPrefix),e=L(r.toString(16)),F(t,e),I(t,Ft);case 1:n.status=2;var i=!0,o=n.chunks,s=0;n=n.children;for(var c=0;c<n.length;c++){for(i=n[c];s<i.index;s++)F(t,o[s]);i=La(e,t,i,r)}for(;s<o.length-1;s++)F(t,o[s]);return s<o.length&&(i=I(t,o[s])),i;case 3:return!0;default:throw Error(a(390))}}var Ia=0;function La(e,t,n,r){var i=n.boundary;if(i===null)return Fa(e,t,n,r);if(i.parentFlushed=!0,i.status===4){var o=i.row;o!==null&&--o.pendingTasks===0&&ia(e,o),i=i.errorDigest,I(t,Y),F(t,Vt),i&&(F(t,Ut),F(t,L(V(i))),F(t,Ht)),I(t,Wt),Fa(e,t,n,r)}else if(i.status!==1)i.status===0&&(i.rootSegmentID=e.nextSegmentId++),0<i.completedSegments.length&&e.partialBoundaries.push(i),Gt(t,e.renderState,i.rootSegmentID),r&&Or(r,i.fallbackState),Fa(e,t,n,r);else if(!Va&&Bi(e,i)&&(Ia+i.byteSize>e.progressiveChunkSize||kr(i.contentState)))i.rootSegmentID=e.nextSegmentId++,e.completedBoundaries.push(i),Gt(t,e.renderState,i.rootSegmentID),Fa(e,t,n,r);else{if(Ia+=i.byteSize,r&&Or(r,i.contentState),n=i.row,n!==null&&Bi(e,i)&&--n.pendingTasks===0&&ia(e,n),I(t,Lt),n=i.completedSegments,n.length!==1)throw Error(a(391));La(e,t,n[0],r)}return I(t,Bt)}function Ra(e,t,n,r){return mn(t,e.renderState,n.parentFormatContext,n.id),La(e,t,n,r),hn(t,n.parentFormatContext)}function za(e,t,n){Ia=n.byteSize;for(var r=n.completedSegments,i=0;i<r.length;i++)Ba(e,t,n,r[i]);r.length=0,r=n.row,r!==null&&Bi(e,n)&&--r.pendingTasks===0&&ia(e,r),Kn(t,n.contentState,e.renderState),r=e.resumableState,e=e.renderState,i=n.rootSegmentID,n=n.contentState;var a=e.stylesToHoist;return e.stylesToHoist=!1,F(t,e.startInlineScript),F(t,st),a?(!(r.instructions&4)&&(r.instructions|=4,F(t,On)),!(r.instructions&2)&&(r.instructions|=2,F(t,bn)),r.instructions&8?F(t,Cn):(r.instructions|=8,F(t,Sn))):(!(r.instructions&2)&&(r.instructions|=2,F(t,bn)),F(t,xn)),r=L(i.toString(16)),F(t,e.boundaryPrefix),F(t,r),F(t,wn),F(t,e.segmentPrefix),F(t,r),a?(F(t,Tn),ur(t,n)):F(t,En),n=I(t,Dn),Nt(t,e)&&n}function Ba(e,t,n,r){if(r.status===2)return!0;var i=n.contentState,o=r.id;if(o===-1){if((r.id=n.rootSegmentID)===-1)throw Error(a(392));return Ra(e,t,r,i)}return o===n.rootSegmentID?Ra(e,t,r,i):(Ra(e,t,r,i),n=e.resumableState,e=e.renderState,F(t,e.startInlineScript),F(t,st),n.instructions&1?F(t,_n):(n.instructions|=1,F(t,gn)),F(t,e.segmentPrefix),o=L(o.toString(16)),F(t,o),F(t,vn),F(t,e.placeholderPrefix),F(t,o),t=I(t,yn),t)}var Va=!1;function Ha(e,t){N=new Uint8Array(2048),P=0;try{if(!(0<e.pendingRootTasks)){var n,r=e.completedRootSegment;if(r!==null){if(r.status===5)return;var i=e.completedPreambleSegments;if(i===null)return;Ia=e.byteSize;var a=e.resumableState,o=e.renderState,s=o.preamble,c=s.htmlChunks,l=s.headChunks,u;if(c){for(u=0;u<c.length;u++)F(t,c[u]);if(l)for(u=0;u<l.length;u++)F(t,l[u]);else F(t,K(`head`)),F(t,st)}else if(l)for(u=0;u<l.length;u++)F(t,l[u]);var d=o.charsetChunks;for(u=0;u<d.length;u++)F(t,d[u]);d.length=0,o.preconnects.forEach(qn,t),o.preconnects.clear();var f=o.viewportChunks;for(u=0;u<f.length;u++)F(t,f[u]);f.length=0,o.fontPreloads.forEach(qn,t),o.fontPreloads.clear(),o.highImagePreloads.forEach(qn,t),o.highImagePreloads.clear(),be=o,o.styles.forEach(tr,t),be=null;var p=o.importMapChunks;for(u=0;u<p.length;u++)F(t,p[u]);p.length=0,o.bootstrapScripts.forEach(qn,t),o.scripts.forEach(qn,t),o.scripts.clear(),o.bulkPreloads.forEach(qn,t),o.bulkPreloads.clear(),c||l||(a.instructions|=32);var m=o.hoistableChunks;for(u=0;u<m.length;u++)F(t,m[u]);for(a=m.length=0;a<i.length;a++){var h=i[a];for(o=0;o<h.length;o++)La(e,t,h[o],null)}var g=e.renderState.preamble,_=g.headChunks;(g.htmlChunks||_)&&F(t,jt(`head`));var v=g.bodyChunks;if(v)for(i=0;i<v.length;i++)F(t,v[i]);La(e,t,r,null),e.completedRootSegment=null;var y=e.renderState;if(e.allPendingTasks!==0||e.clientRenderedBoundaries.length!==0||e.completedBoundaries.length!==0||e.trackedPostpones!==null&&(e.trackedPostpones.rootNodes.length!==0||e.trackedPostpones.rootSlots!==null)){var b=e.resumableState;if(!(b.instructions&64)){if(b.instructions|=64,F(t,y.startInlineScript),!(b.instructions&32)){b.instructions|=32;var x=`_`+b.idPrefix+`R_`;F(t,ir),F(t,L(V(x))),F(t,Xe)}F(t,st),F(t,q),I(t,Se)}}Nt(t,y)}var S=e.renderState;r=0;var C=S.viewportChunks;for(r=0;r<C.length;r++)F(t,C[r]);C.length=0,S.preconnects.forEach(qn,t),S.preconnects.clear(),S.fontPreloads.forEach(qn,t),S.fontPreloads.clear(),S.highImagePreloads.forEach(qn,t),S.highImagePreloads.clear(),S.styles.forEach(rr,t),S.scripts.forEach(qn,t),S.scripts.clear(),S.bulkPreloads.forEach(qn,t),S.bulkPreloads.clear();var w=S.hoistableChunks;for(r=0;r<w.length;r++)F(t,w[r]);w.length=0;var T=e.clientRenderedBoundaries;for(n=0;n<T.length;n++){var E=T[n];S=t;var D=e.resumableState,O=e.renderState,k=E.rootSegmentID,A=E.errorDigest;F(S,O.startInlineScript),F(S,st),D.instructions&4?F(S,An):(D.instructions|=4,F(S,kn)),F(S,O.boundaryPrefix),F(S,L(k.toString(16))),F(S,jn),A&&(F(S,Mn),F(S,L(Fn(A||``))));var j=I(S,Nn);if(!j){e.destination=null,n++,T.splice(0,n);return}}T.splice(0,n);var M=e.completedBoundaries;for(n=0;n<M.length;n++)if(!za(e,t,M[n])){e.destination=null,n++,M.splice(0,n);return}M.splice(0,n),ee(t),N=new Uint8Array(2048),P=0,Va=!0;var te=e.partialBoundaries;for(n=0;n<te.length;n++){var R=te[n];a:{T=e,E=t,Ia=R.byteSize;var ne=R.completedSegments;for(j=0;j<ne.length;j++)if(!Ba(T,E,R,ne[j])){j++,ne.splice(0,j);var re=!1;break a}ne.splice(0,j);var ie=R.row;ie!==null&&ie.together&&R.pendingTasks===1&&(ie.pendingTasks===1?aa(T,ie,ie.hoistables):ie.pendingTasks--),re=Kn(E,R.contentState,T.renderState)}if(!re){e.destination=null,n++,te.splice(0,n);return}}te.splice(0,n),Va=!1;var z=e.completedBoundaries;for(n=0;n<z.length;n++)if(!za(e,t,z[n])){e.destination=null,n++,z.splice(0,n);return}z.splice(0,n)}}finally{Va=!1,e.allPendingTasks===0&&e.clientRenderedBoundaries.length===0&&e.completedBoundaries.length===0?(e.flushScheduled=!1,n=e.resumableState,n.hasBody&&F(t,jt(`body`)),n.hasHtml&&F(t,jt(`html`)),ee(t),e.status=14,t.close(),e.destination=null):ee(t)}}function Ua(e){e.flushScheduled=e.destination!==null,M(function(){return ja(e)}),k(function(){e.status===10&&(e.status=11),e.trackedPostpones===null&&Ta(e,e.pendingRootTasks===0)})}function Wa(e){!1===e.flushScheduled&&e.pingedTasks.length===0&&e.destination!==null&&(e.flushScheduled=!0,k(function(){var t=e.destination;t?Ha(e,t):e.flushScheduled=!1}))}function Ga(e,t){if(e.status===13)e.status=14,re(t,e.fatalError);else if(e.status!==14&&e.destination===null){e.destination=t;try{Ha(e,t)}catch(t){na(e,t,{}),ra(e,t)}}}function Ka(e,t){(e.status===11||e.status===10)&&(e.status=12);try{var n=e.abortableTasks;if(0<n.size){var r=t===void 0?Error(a(432)):typeof t==`object`&&t&&typeof t.then==`function`?Error(a(530)):t;e.fatalError=r,n.forEach(function(t){return wa(t,e,r)}),n.clear()}e.destination!==null&&Ha(e,e.destination)}catch(t){na(e,t,{}),ra(e,t)}}function qa(e,t,n){if(t===null)n.rootNodes.push(e);else{var r=n.workingMap,i=r.get(t);i===void 0&&(i=[t[1],t[2],[],null],r.set(t,i),qa(i,t[0],n)),i[2].push(e)}}function Ja(e){var t=e.trackedPostpones;if(t===null||t.rootNodes.length===0&&t.rootSlots===null)return e.trackedPostpones=null;if(e.completedRootSegment===null||e.completedRootSegment.status!==5&&e.completedPreambleSegments!==null){var n=e.nextSegmentId,r=t.rootSlots,i=e.resumableState;i.bootstrapScriptContent=void 0,i.bootstrapScripts=void 0,i.bootstrapModules=void 0}else{n=0,r=-1,i=e.resumableState;var a=e.renderState;i.nextFormID=0,i.hasBody=!1,i.hasHtml=!1,i.unknownResources={font:a.resets.font},i.dnsResources=a.resets.dns,i.connectResources=a.resets.connect,i.imageResources=a.resets.image,i.styleResources=a.resets.style,i.scriptResources={},i.moduleUnknownResources={},i.moduleScriptResources={},i.instructions=0}return{nextSegmentId:n,rootFormatContext:e.rootFormatContext,progressiveChunkSize:e.progressiveChunkSize,resumableState:e.resumableState,replayNodes:t.rootNodes,replaySlots:r}}function Ya(){var e=t.version;if(e!==`19.2.4`)throw Error(a(527,e,`19.2.4`))}Ya(),Ya(),e.prerender=function(e,t){return new Promise(function(n,r){var i=t?t.onHeaders:void 0,a;i&&(a=function(e){i(new Headers(e))});var o=Ne(t?t.identifierPrefix:void 0,t?t.unstable_externalRuntimeSrc:void 0,t?t.bootstrapScriptContent:void 0,t?t.bootstrapScripts:void 0,t?t.bootstrapModules:void 0),s=Wi(e,o,Me(o,void 0,t?t.unstable_externalRuntimeSrc:void 0,t?t.importMap:void 0,a,t?t.maxHeadersLength:void 0),Ie(t?t.namespaceURI:void 0),t?t.progressiveChunkSize:void 0,t?t.onError:void 0,function(){var e=new ReadableStream({type:`bytes`,pull:function(e){Ga(s,e)},cancel:function(e){s.destination=null,Ka(s,e)}},{highWaterMark:0});e={postponed:Ja(s),prelude:e},n(e)},void 0,void 0,r,t?t.onPostpone:void 0);if(t&&t.signal){var c=t.signal;if(c.aborted)Ka(s,c.reason);else{var l=function(){Ka(s,c.reason),c.removeEventListener(`abort`,l)};c.addEventListener(`abort`,l)}}Ua(s)})},e.renderToReadableStream=function(e,t){return new Promise(function(n,r){var i,a,o=new Promise(function(e,t){a=e,i=t}),s=t?t.onHeaders:void 0,c;s&&(c=function(e){s(new Headers(e))});var l=Ne(t?t.identifierPrefix:void 0,t?t.unstable_externalRuntimeSrc:void 0,t?t.bootstrapScriptContent:void 0,t?t.bootstrapScripts:void 0,t?t.bootstrapModules:void 0),u=Ui(e,l,Me(l,t?t.nonce:void 0,t?t.unstable_externalRuntimeSrc:void 0,t?t.importMap:void 0,c,t?t.maxHeadersLength:void 0),Ie(t?t.namespaceURI:void 0),t?t.progressiveChunkSize:void 0,t?t.onError:void 0,a,function(){var e=new ReadableStream({type:`bytes`,pull:function(e){Ga(u,e)},cancel:function(e){u.destination=null,Ka(u,e)}},{highWaterMark:0});e.allReady=o,n(e)},function(e){o.catch(function(){}),r(e)},i,t?t.onPostpone:void 0,t?t.formState:void 0);if(t&&t.signal){var d=t.signal;if(d.aborted)Ka(u,d.reason);else{var f=function(){Ka(u,d.reason),d.removeEventListener(`abort`,f)};d.addEventListener(`abort`,f)}}Ua(u)})},e.resume=function(e,t,n){return new Promise(function(r,i){var a,o,s=new Promise(function(e,t){o=e,a=t}),c=Gi(e,t,Me(t.resumableState,n?n.nonce:void 0,void 0,void 0,void 0,void 0),n?n.onError:void 0,o,function(){var e=new ReadableStream({type:`bytes`,pull:function(e){Ga(c,e)},cancel:function(e){c.destination=null,Ka(c,e)}},{highWaterMark:0});e.allReady=s,r(e)},function(e){s.catch(function(){}),i(e)},a,n?n.onPostpone:void 0);if(n&&n.signal){var l=n.signal;if(l.aborted)Ka(c,l.reason);else{var u=function(){Ka(c,l.reason),l.removeEventListener(`abort`,u)};l.addEventListener(`abort`,u)}}Ua(c)})},e.resumeAndPrerender=function(e,t,n){return new Promise(function(r,i){var a=Ki(e,t,Me(t.resumableState,void 0,void 0,void 0,void 0,void 0),n?n.onError:void 0,function(){var e=new ReadableStream({type:`bytes`,pull:function(e){Ga(a,e)},cancel:function(e){a.destination=null,Ka(a,e)}},{highWaterMark:0});e={postponed:Ja(a),prelude:e},r(e)},void 0,void 0,i,n?n.onPostpone:void 0);if(n&&n.signal){var o=n.signal;if(o.aborted)Ka(a,o.reason);else{var s=function(){Ka(a,o.reason),o.removeEventListener(`abort`,s)};o.addEventListener(`abort`,s)}}Ua(a)})},e.version=`19.2.4`})),wl=t((e=>{var t=Sl(),n=Cl();e.version=t.version,e.renderToString=t.renderToString,e.renderToStaticMarkup=t.renderToStaticMarkup,e.renderToReadableStream=n.renderToReadableStream,e.resume=n.resume}))(),Z={AIR:0,GRASS:1,DIRT:2,PATH:3,PLAZA:4,SOIL:5,PLANK:6,TIMBER:7,PLASTER:8,STONE:9,ROOF_RED:10,ROOF_DARK:11,TRUNK:12,LEAVES:13,FENCE:14,MEADOW:15,FLOWERS:16,WINDOW:17,POST:18,LAMP:19,WATER:20,ROOF_SLATE:21,ROOF_MOSS:22,WALL_SAGE:23},Q={GRASS_TOP:0,GRASS_SIDE:1,DIRT:2,PATH:3,PLAZA:4,SOIL:5,PLANK:6,TIMBER:7,PLASTER:8,STONE:9,ROOF_RED:10,ROOF_DARK:11,TRUNK_SIDE:12,TRUNK_TOP:13,LEAVES:14,FENCE:15,MEADOW:16,FLOWERS:17,WINDOW:18,POST:19,LAMP:20,WATER:21,ROOF_SLATE:22,ROOF_MOSS:23,WALL_SAGE:24},Tl={[Z.GRASS]:[Q.GRASS_TOP,Q.GRASS_SIDE,Q.DIRT],[Z.DIRT]:[Q.DIRT,Q.DIRT,Q.DIRT],[Z.PATH]:[Q.PATH,Q.DIRT,Q.DIRT],[Z.PLAZA]:[Q.PLAZA,Q.PLAZA,Q.PLAZA],[Z.SOIL]:[Q.SOIL,Q.DIRT,Q.DIRT],[Z.PLANK]:[Q.PLANK,Q.PLANK,Q.PLANK],[Z.TIMBER]:[Q.TIMBER,Q.TIMBER,Q.TIMBER],[Z.PLASTER]:[Q.PLASTER,Q.PLASTER,Q.PLASTER],[Z.STONE]:[Q.STONE,Q.STONE,Q.STONE],[Z.ROOF_RED]:[Q.ROOF_RED,Q.ROOF_RED,Q.ROOF_RED],[Z.ROOF_DARK]:[Q.ROOF_DARK,Q.ROOF_DARK,Q.ROOF_DARK],[Z.TRUNK]:[Q.TRUNK_TOP,Q.TRUNK_SIDE,Q.TRUNK_TOP],[Z.LEAVES]:[Q.LEAVES,Q.LEAVES,Q.LEAVES],[Z.FENCE]:[Q.FENCE,Q.FENCE,Q.FENCE],[Z.MEADOW]:[Q.MEADOW,Q.GRASS_SIDE,Q.DIRT],[Z.FLOWERS]:[Q.FLOWERS,Q.GRASS_SIDE,Q.DIRT],[Z.WINDOW]:[Q.WINDOW,Q.WINDOW,Q.WINDOW],[Z.POST]:[Q.POST,Q.POST,Q.POST],[Z.LAMP]:[Q.LAMP,Q.LAMP,Q.LAMP],[Z.WATER]:[Q.WATER,Q.WATER,Q.WATER],[Z.ROOF_SLATE]:[Q.ROOF_SLATE,Q.ROOF_SLATE,Q.ROOF_SLATE],[Z.ROOF_MOSS]:[Q.ROOF_MOSS,Q.ROOF_MOSS,Q.ROOF_MOSS],[Z.WALL_SAGE]:[Q.WALL_SAGE,Q.WALL_SAGE,Q.WALL_SAGE]},El=e=>e!==Z.AIR&&e!==Z.WATER,Dl=e=>e===Z.LAMP;function Ol(e){return()=>{e|=0,e=e+1831565813|0;let t=Math.imul(e^e>>>15,1|e);return t=t+Math.imul(t^t>>>7,61|t)^t,((t^t>>>14)>>>0)/4294967296}}function kl(e,t,n,r){for(let i=0;i<16;i++)for(let a=0;a<16;a++){let o=1+(t()-.5)*r;e.fillStyle=`rgb(${n[0]*o|0},${n[1]*o|0},${n[2]*o|0})`,e.fillRect(a,i,1,1)}}var Al=8;function jl(){let e=document.createElement(`canvas`);e.width=e.height=Al*16;let t=e.getContext(`2d`),n=(e,n)=>{let r=Ol(5610^e*7919);t.save(),t.translate(e%Al*16,Math.floor(e/Al)*16),n(t,r),t.restore()};n(Q.GRASS_TOP,(e,t)=>{kl(e,t,[154,186,118],.16);for(let n=0;n<10;n++)e.fillStyle=`rgba(110,146,82,.5)`,e.fillRect(t()*16|0,t()*16|0,1,1)}),n(Q.DIRT,(e,t)=>{kl(e,t,[158,126,94],.2);for(let n=0;n<6;n++)e.fillStyle=`rgba(110,84,58,.6)`,e.fillRect(t()*16|0,t()*16|0,2,1)}),n(Q.GRASS_SIDE,(e,t)=>{kl(e,t,[158,126,94],.2);for(let n=0;n<16;n++){let r=3+(t()*3|0);for(let i=0;i<r;i++){let r=1+(t()-.5)*.2;e.fillStyle=`rgb(${150*r|0},${182*r|0},${114*r|0})`,e.fillRect(n,i,1,1)}}}),n(Q.PATH,(e,t)=>{kl(e,t,[205,176,132],.13);for(let n=0;n<7;n++)e.fillStyle=`rgba(170,140,100,.55)`,e.fillRect(t()*15|0,t()*15|0,1+(t()*2|0),1)}),n(Q.PLAZA,(e,t)=>{kl(e,t,[196,180,152],.1);for(let n=0;n<8;n++){let n=t()*12|0,r=t()*12|0,i=2+(t()*3|0),a=2+(t()*2|0),o=.88+t()*.24;e.fillStyle=`rgb(${200*o|0},${184*o|0},${156*o|0})`,e.fillRect(n,r,i,a),e.strokeStyle=`rgba(120,106,84,.55)`,e.strokeRect(n+.5,r+.5,i-1,a-1)}}),n(Q.SOIL,(e,t)=>{kl(e,t,[126,96,66],.16);for(let t=1;t<16;t+=4)e.fillStyle=`rgba(84,60,40,.7)`,e.fillRect(0,t,16,1);for(let n=0;n<4;n++)e.fillStyle=`rgba(150,190,120,.7)`,e.fillRect(t()*15|0,t()*15|0,1,1)}),n(Q.PLANK,(e,t)=>{for(let n=0;n<16;n++)for(let r=0;r<16;r++){let i=1+(t()-.5)*.12,a=n%4==3||(r===7||r===15)&&Math.floor(n/4)%2==0||(r===3||r===11)&&Math.floor(n/4)%2==1?.6:1;e.fillStyle=`rgb(${206*i*a|0},${170*i*a|0},${118*i*a|0})`,e.fillRect(r,n,1,1)}}),n(Q.TIMBER,(e,t)=>{for(let n=0;n<16;n++){let r=n%4==0?.68:t()<.25?.85:1;for(let i=0;i<16;i++){let a=(1+(t()-.5)*.15)*r;e.fillStyle=`rgb(${150*a|0},${114*a|0},${78*a|0})`,e.fillRect(n,i,1,1)}}}),n(Q.PLASTER,(e,t)=>{kl(e,t,[235,226,206],.06);for(let n=0;n<5;n++)e.fillStyle=`rgba(196,182,152,.5)`,e.fillRect(t()*16|0,t()*16|0,1,1)}),n(Q.STONE,(e,t)=>{kl(e,t,[172,166,152],.12);for(let n=0;n<7;n++){let n=t()*13|0,r=t()*13|0,i=2+(t()*3|0),a=2+(t()*2|0),o=.82+t()*.4;e.fillStyle=`rgb(${176*o|0},${170*o|0},${156*o|0})`,e.fillRect(n,r,i,a),e.strokeStyle=`rgba(104,98,86,.6)`,e.strokeRect(n+.5,r+.5,i-1,a-1)}}),n(Q.ROOF_RED,(e,t)=>{for(let n=0;n<16;n++)for(let r=0;r<16;r++){let i=Math.floor(n/4),a=n%4==3||(r+(i%2==0?0:4))%8==7,o=(1+(t()-.5)*.14)*(a?.62:1);e.fillStyle=`rgb(${206*o|0},${118*o|0},${86*o|0})`,e.fillRect(r,n,1,1)}}),n(Q.ROOF_DARK,(e,t)=>{for(let n=0;n<16;n++)for(let r=0;r<16;r++){let i=Math.floor(n/4),a=n%4==3||(r+(i%2==0?0:4))%8==7,o=(1+(t()-.5)*.14)*(a?.6:1);e.fillStyle=`rgb(${118*o|0},${104*o|0},${140*o|0})`,e.fillRect(r,n,1,1)}});let r=e=>(t,n)=>{for(let r=0;r<16;r++)for(let i=0;i<16;i++){let a=Math.floor(r/4),o=r%4==3||(i+(a%2==0?0:4))%8==7,s=(1+(n()-.5)*.14)*(o?.62:1);t.fillStyle=`rgb(${e[0]*s|0},${e[1]*s|0},${e[2]*s|0})`,t.fillRect(i,r,1,1)}};n(Q.ROOF_SLATE,r([110,128,146])),n(Q.ROOF_MOSS,r([106,136,84])),n(Q.WALL_SAGE,(e,t)=>{kl(e,t,[205,214,178],.07);for(let n=0;n<5;n++)e.fillStyle=`rgba(160,172,138,.5)`,e.fillRect(t()*16|0,t()*16|0,1,1)}),n(Q.TRUNK_SIDE,(e,t)=>{for(let n=0;n<16;n++){let r=n%5==0?.7:1;for(let i=0;i<16;i++){let a=(1+(t()-.5)*.18)*r;e.fillStyle=`rgb(${138*a|0},${106*a|0},${72*a|0})`,e.fillRect(n,i,1,1)}}}),n(Q.TRUNK_TOP,(e,t)=>{kl(e,t,[172,138,96],.12),e.strokeStyle=`rgba(110,82,52,.85)`;for(let t=1;t<4;t++)e.strokeRect(t+.5,t+.5,15-t*2,15-t*2)}),n(Q.LEAVES,(e,t)=>{for(let n=0;n<16;n++)for(let r=0;r<16;r++){let i=t();e.fillStyle=i<.1?`rgba(70,102,52,1)`:`rgb(${112+i*34|0},${158+i*34|0},${88+i*26|0})`,e.fillRect(r,n,1,1)}}),n(Q.FENCE,(e,t)=>{kl(e,t,[96,116,78],.2);for(let n=0;n<16;n+=4)for(let r=0;r<16;r++){let i=1+(t()-.5)*.1;e.fillStyle=`rgb(${226*i|0},${206*i|0},${164*i|0})`,e.fillRect(n,r,2,1)}e.fillStyle=`rgba(190,168,128,.9)`,e.fillRect(0,3,16,1),e.fillRect(0,10,16,1)}),n(Q.MEADOW,(e,t)=>{kl(e,t,[168,198,126],.14);for(let n=0;n<12;n++){e.fillStyle=`rgba(126,162,90,.8)`;let n=t()*16|0,r=t()*14|0;e.fillRect(n,r,1,2)}}),n(Q.FLOWERS,(e,t)=>{kl(e,t,[154,186,118],.16);let n=[`#e8a04b`,`#e08a6d`,`#f3f7d4`,`#bb9dd4`];for(let r=0;r<7;r++)e.fillStyle=n[t()*n.length|0],e.fillRect(1+(t()*14|0),1+(t()*14|0),1,1)}),n(Q.WINDOW,(e,t)=>{kl(e,t,[255,217,138],.1),e.fillStyle=`rgba(255,240,200,.8)`,e.fillRect(2,2,5,5),e.strokeStyle=`#8a6a48`,e.lineWidth=2,e.strokeRect(1,1,14,14),e.fillStyle=`#8a6a48`,e.fillRect(7,0,2,16),e.fillRect(0,7,16,2)}),n(Q.POST,(e,t)=>{for(let n=0;n<16;n++){let r=n%6==0?.72:1;for(let i=0;i<16;i++){let a=(1+(t()-.5)*.14)*r;e.fillStyle=`rgb(${104*a|0},${82*a|0},${56*a|0})`,e.fillRect(n,i,1,1)}}}),n(Q.LAMP,(e,t)=>{kl(e,t,[255,226,158],.06),e.fillStyle=`rgba(255,244,208,.95)`,e.fillRect(4,4,8,8),e.strokeStyle=`rgba(120,94,58,.8)`,e.strokeRect(.5,.5,15,15)}),n(Q.WATER,(e,t)=>{kl(e,t,[98,152,176],.12);for(let n=0;n<5;n++)e.fillStyle=`rgba(190,224,238,.55)`,e.fillRect(t()*10|0,t()*16|0,4+(t()*5|0),1)});let i=new Hi(e);return i.magFilter=A,i.minFilter=A,i.generateMipmaps=!1,i.colorSpace=at,{texture:i,uvFor:e=>[e%Al/Al,1-(Math.floor(e/Al)+1)/Al]}}var Ml=1/Al,Nl=20;Nl/5;var Pl=50,Fl=Pl*4;Fl+32,Fl+32;var Il=(e,t,n)=>(t*232+n)*232+e;function Ll(e){let t=Math.max(1,e.w||1),n=Math.max(1,e.h||1),r=Math.floor(e.x/Nl),i=Math.floor(e.y/Nl);return[Math.min(Math.max(1,r-Math.floor(t/2)),Pl-1-t),Math.min(Math.max(1,i-Math.floor(n/2)),Pl-1-n),t,n]}function Rl(e){return[Math.min(Math.max(0,Math.floor(e[0]/Nl)),Pl-2),Math.min(Math.max(0,Math.floor(e[1]/Nl)),Pl-2)]}var zl={square:`plaza`,library:`library`,workshop:`workshop`,garden:`garden`,well:`well`,meadow:`meadow`,"old-bench":`bench`},Bl={read:`library`,create:`workshop`,gather:`pavilion`,trade:`stall`,tend:`garden`,play:`pond`,rest:`bench`,remember:`cairn`},Vl=e=>zl[e.id]||Bl[(e.affordances||[])[0]||``]||`cottage`,Hl={ember:Z.ROOF_RED,slate:Z.ROOF_SLATE,moss:Z.ROOF_MOSS,dusk:Z.ROOF_DARK},Ul={plaster:Z.PLASTER,timber:Z.TIMBER,sage:Z.WALL_SAGE};function Wl(e,t,n){let r=e*374761393+t*668265263+n*974711|0;return r=Math.imul(r^r>>>13,1274126177),((r^r>>>16)>>>0)/4294967296}function Gl(e){let t=new Uint8Array(5568*232),n=[],r=[],i=(e,n,r,i)=>{e>=0&&e<232&&n>=0&&n<24&&r>=0&&r<232&&(t[Il(e,n,r)]=i)},a=(e,n,r)=>{if(n<0)return Z.STONE;if(n>=24)return Z.AIR;let i=Math.floor(e),a=Math.floor(n),o=Math.floor(r);return i<0||i>=232||o<0||o>=232?Z.STONE:t[Il(i,a,o)]};for(let e=0;e<232;e++)for(let t=0;t<232;t++)i(t,0,e,Z.DIRT),i(t,1,e,Z.DIRT),i(t,2,e,Z.GRASS);for(let e=2;e<230;e++)for(let t=2;t<230;t++)t>=16&&t<216&&e>=16&&e<216||Wl(t,e,77)<.02&&f(t,e,3+(Wl(t,e,78)*2|0));let o=new Set;for(let t of e.roads??[])o.add(`${t[0]},${t[1]}`),d(t[0],t[1],1,1,(e,t)=>i(e,2,t,Z.PATH));for(let t of e.places){let[e,a,o,s]=Ll(t),c=e*4+16,d=a*4+16,f=c+o*4-1,g=d+s*4-1;n.push({x0:c,z0:d,x1:f,z1:g,name:`the ${t.name.replace(/^the /i,``)}`});let _=l(t);switch(_&&r.push({placeId:t.id,placeName:`the ${t.name.replace(/^the /i,``)}`,label:_.label,bx:c+1,bz:d+1}),Vl(t)){case`plaza`:u(c,d,f,g,(e,t)=>i(e,2,t,Z.PLAZA));break;case`garden`:u(c,d,f,g,(e,t)=>i(e,2,t,(t-d)%2==0?Z.SOIL:Z.FLOWERS)),p(c,d,f,g,i);break;case`meadow`:u(c,d,f,g,(e,t)=>i(e,2,t,Wl(e,t,9)<.18?Z.FLOWERS:Z.MEADOW));break;case`pond`:{u(c,d,f,g,(e,t)=>i(e,2,t,Z.MEADOW));let e=c+f>>1,t=d+g>>1;u(e-2,t-1,e+2,t+1,(e,t)=>i(e,2,t,Z.WATER));break}case`well`:{let e=c+f>>1,t=d+g>>1;u(e-1,t-1,e+2,t+2,(n,r)=>{n===e-1||n===e+2||r===t-1||r===t+2?i(n,3,r,Z.STONE):i(n,2,r,Z.WATER)});for(let[n,r]of[[e-1,t-1],[e+2,t-1],[e-1,t+2],[e+2,t+2]])i(n,4,r,Z.POST),i(n,5,r,Z.POST);u(e-1,t-1,e+2,t+2,(e,t)=>i(e,6,t,Z.ROOF_RED));break}case`bench`:{let e=c+f>>1,t=d+g>>1;i(e,3,t,Z.PLANK),i(e+1,3,t,Z.PLANK);break}case`cairn`:{let e=c+f>>1,t=d+g>>1;i(e,3,t,Z.STONE),i(e+1,3,t,Z.STONE),i(e,3,t+1,Z.STONE),i(e,4,t,Z.STONE);break}case`stall`:for(let[e,t]of[[c,d],[f,d],[c,g],[f,g]])i(e,3,t,Z.POST),i(e,4,t,Z.POST),i(e,5,t,Z.POST);for(let e=c;e<=f;e++)i(e,3,g,Z.PLANK);u(c,d,f,g,(e,t)=>i(e,6,t,Z.ROOF_RED));break;case`pavilion`:u(c,d,f,g,(e,t)=>i(e,2,t,Z.PLAZA));for(let[e,t]of[[c,d],[f,d],[c,g],[f,g]])for(let n=3;n<=5;n++)i(e,n,t,Z.POST);m(c,d,f,g,6,Z.ROOF_RED,2,i);break;case`library`:h(c,d,f,g,{wall:Z.TIMBER,corner:Z.TIMBER,roof:Z.ROOF_DARK,wallH:5},t,i);break;case`workshop`:h(c,d,f,g,{wall:Z.PLANK,corner:Z.TIMBER,roof:Z.ROOF_RED,wallH:4},t,i);break;default:h(c,d,f,g,{wall:Z.PLASTER,corner:Z.TIMBER,roof:Z.ROOF_RED,wallH:4},t,i)}}for(let t of e.beings){if(t.kind===`visitor`||!t.home_xy)continue;let[e,r]=Rl(t.home_xy),a=e*4+16,s=r*4+16,c=a+8-1,l=s+8-1;n.push({x0:a,z0:s,x1:c,z1:l,name:t.home_name?`“${t.home_name}” — ${t.name}'s home`:`${t.name}'s home`});let u=null,d=1/0,f=(e+1)*4,p=(r+1)*4;for(let e of o){let[t,n]=e.split(`,`).map(Number),r=Math.abs(t*4+2-f)+Math.abs(n*4+2-p);r<d&&(d=r,u=[t*4+2,n*4+2])}let m=u?Math.abs(u[0]-f)>=Math.abs(u[1]-p)?u[0]>f?`e`:`w`:u[1]>p?`s`:`n`:`e`,h=t.home_look||{},_=Hl[h.roof||``]??Z.ROOF_RED;g(a,s,c,l,m,i,Ul[h.wall||``]??Z.PLASTER,_)}for(let t of e.props??[]){let e=t.tile[0]*4+16,n=t.tile[1]*4+16,r=e+1,o=n+1;t.kind===`tree`?f(r,o,3+(Wl(r,o,5)*2|0)):t.kind===`bush`?(i(r,3,o,Z.LEAVES),i(r+1,3,o,Z.LEAVES),i(r,3,o+1,Z.LEAVES)):t.kind===`flowers`?d(t.tile[0],t.tile[1],1,1,(e,t)=>{Wl(e,t,6)<.5&&i(e,2,t,Z.FLOWERS)}):t.kind===`lamp`&&a(e,2,n)!==Z.AIR&&(i(e,3,n,Z.POST),i(e,4,n,Z.POST),i(e,5,n,Z.LAMP))}for(let t of e.objects??[]){let e=t.tile[0]*4+16,r=t.tile[1]*4+16,a=e+1,o=r+1;if(t.staked){i(a,2,o,Z.SOIL),i(a+1,2,o,Z.SOIL),i(a,2,o+1,Z.SOIL),n.push({x0:e,z0:r,x1:e+4-1,z1:r+4-1,name:`a beginning — a ${t.kind}`});continue}switch(n.push({x0:e,z0:r,x1:e+4-1,z1:r+4-1,name:`“${t.name}”`}),t.kind){case`bench`:i(a,3,o,Z.PLANK),i(a+1,3,o,Z.PLANK);break;case`signpost`:i(a,3,o,Z.POST),i(a,4,o,Z.POST),i(a,5,o,Z.PLANK);break;case`planter`:i(a,3,o,Z.PLANK),i(a+1,3,o,Z.PLANK),i(a,4,o,Z.LEAVES),i(a+1,4,o,Z.FLOWERS);break;case`lantern`:i(a,3,o,Z.POST),i(a,4,o,Z.POST),i(a,5,o,Z.LAMP);break;case`cairn`:i(a,3,o,Z.STONE),i(a+1,3,o,Z.STONE),i(a,3,o+1,Z.STONE),i(a,4,o,Z.STONE);break;case`sculpture`:i(a,3,o,Z.STONE),i(a,4,o,Z.STONE),i(a,5,o,Z.STONE),i(a+1,3,o,Z.STONE);break;case`fountain`:u(a-1,o-1,a+2,o+2,(e,t)=>{e===a-1||e===a+2||t===o-1||t===o+2?i(e,3,t,Z.STONE):i(e,2,t,Z.WATER)});break;case`shrine`:i(a,3,o,Z.STONE),i(a,4,o,Z.STONE),i(a+1,3,o,Z.STONE),i(a+1,4,o,Z.STONE),i(a,5,o,Z.ROOF_RED),i(a+1,5,o,Z.ROOF_RED),i(a,3,o+1,Z.LAMP);break;default:i(a,3,o,Z.STONE)}}let s=e.places.find(e=>e.id===`square`)||e.places.find(e=>(e.affordances||[]).includes(`gather`))||e.places[0],c={x:232/2,y:3,z:124,yaw:0};if(s){let[e,t,n,r]=Ll(s);c={x:(e+n/2)*4+16,y:3,z:(t+r)*4+16+5,yaw:0}}return{blocks:t,labels:n,lecterns:r,spawn:c,get:a};function u(e,t,n,r,i){for(let a=t;a<=r;a++)for(let t=e;t<=n;t++)i(t,a)}function d(e,t,n,r,i){u(e*4+16,t*4+16,(e+n)*4+16-1,(t+r)*4+16-1,i)}function f(e,t,n){for(let r=0;r<n;r++)i(e,3+r,t,Z.TRUNK);let r=3+n;for(let n=-1;n<=1;n++)for(let o=-2;o<=2;o++)for(let s=-2;s<=2;s++){let c=n===1?1:2;Math.abs(s)>c||Math.abs(o)>c||Math.abs(s)===c&&Math.abs(o)===c&&Wl(e+s,t+o,8)<.5||a(e+s,r+n,t+o)===Z.AIR&&i(e+s,r+n,t+o,Z.LEAVES)}i(e,r+2,t,Z.LEAVES)}function p(e,t,n,r,i){let a=e+n>>1,o=t+r>>1;for(let o=e;o<=n;o++)Math.abs(o-a)>1&&(i(o,3,t,Z.FENCE),i(o,3,r,Z.FENCE));for(let a=t;a<=r;a++)Math.abs(a-o)>1&&(i(e,3,a,Z.FENCE),i(n,3,a,Z.FENCE))}function m(e,t,n,r,i,a,o,s){let c=e-1,l=t-1,d=n+1,f=r+1;for(let e=0;e<o&&d>=c&&f>=l;e++)u(c,l,d,f,(t,n)=>s(t,i+e,n,a)),c++,l++,d--,f--}function h(e,t,n,r,i,a,o){u(e,t,n,r,(e,t)=>o(e,2,t,Z.PLANK)),u(e,t,n,r,(a,s)=>{if(!(a===e||a===n||s===t||s===r))return;let c=(a===e||a===n)&&(s===t||s===r);for(let e=0;e<i.wallH;e++)o(a,3+e,s,c?i.corner:i.wall)});for(let i=e+2;i<=n-2;i+=4)o(i,4,t,Z.WINDOW),o(i,4,r,Z.WINDOW);for(let i=t+2;i<=r-2;i+=4)o(e,4,i,Z.WINDOW),o(n,4,i,Z.WINDOW);if(a.door_x!=null&&a.door_y!=null){let i=a.door_x*4+16,s=a.door_y*4+16,c=null;s<=t?c=`n`:s+4-1>=r?c=`s`:i<=e?c=`w`:i+4-1>=n&&(c=`e`);let l=(e,t)=>{for(let n=3;n<=5;n++)o(e,n,t,Z.AIR)};c===`n`?(l(i+1,t),l(i+2,t)):c===`s`?(l(i+1,r),l(i+2,r)):c===`w`?(l(e,s+1),l(e,s+2)):c===`e`&&(l(n,s+1),l(n,s+2))}m(e,t,n,r,3+i.wallH,i.roof,4,o)}function g(e,t,n,r,i,a,o=Z.PLASTER,s=Z.ROOF_RED){u(e,t,n,r,(e,t)=>a(e,2,t,Z.PLANK)),u(e,t,n,r,(i,s)=>{if(!(i===e||i===n||s===t||s===r))return;let c=(i===e||i===n)&&(s===t||s===r);for(let e=0;e<3;e++)a(i,3+e,s,c?Z.TIMBER:o)});let c=e+n>>1,l=t+r>>1,d=(e,t)=>{a(e,3,t,Z.AIR),a(e,4,t,Z.AIR)};i===`e`?(d(n,l),d(n,l+1),a(e,4,l,Z.WINDOW)):i===`w`?(d(e,l),d(e,l+1),a(n,4,l,Z.WINDOW)):i===`s`?(d(c,r),d(c+1,r),a(c,4,t,Z.WINDOW)):(d(c,t),d(c+1,t),a(c,4,r,Z.WINDOW)),m(e,t,n,r,6,s,4,a)}}var Kl=5,ql=12,Jl=3.5,Yl=60,Xl=1.5,Zl={infant:.6,child:.8,adolescent:.92},Ql=1.7,$l=new Map;function eu(e,t){let n=`${e}:${t}`,r=$l.get(n);if(r)return r;let i=new Promise((n,r)=>{let i=(0,wl.renderToStaticMarkup)((0,T.createElement)(d,{c:e,p:t,size:48})).replace(`<svg `,`<svg xmlns="http://www.w3.org/2000/svg" `),a=new Image;a.onload=()=>{let e=document.createElement(`canvas`);e.width=168,e.height=224;let t=e.getContext(`2d`),r=document.createElement(`canvas`);r.width=168,r.height=224;let i=r.getContext(`2d`);i.drawImage(a,10,10,148,204),i.globalCompositeOperation=`source-in`,i.fillStyle=`#f2ead2`,i.fillRect(0,0,168,224);for(let e=0;e<8;e++)t.drawImage(r,Math.cos(e*Math.PI/4)*4,Math.sin(e*Math.PI/4)*4);t.drawImage(a,10,10,148,204);let o=new Hi(e);o.colorSpace=at,o.magFilter=N,o.minFilter=N,o.generateMipmaps=!1,n(o)},a.onerror=()=>r(Error(`avatar rasterize failed`)),a.src=`data:image/svg+xml;utf8,`+encodeURIComponent(i)});return $l.set(n,i),i}function tu(e,t){let n=document.createElement(`canvas`);n.width=256,n.height=t?96:64;let r=n.getContext(`2d`);r.font=`600 26px -apple-system, system-ui, sans-serif`;let i=Math.min(236,r.measureText(e).width+28),a=(256-i)/2;if(r.fillStyle=t?`rgba(12,30,45,0.85)`:`rgba(23,20,16,0.82)`,r.beginPath(),r.roundRect(a,12,i,40,20),r.fill(),r.fillStyle=t?`#bae6fd`:`#e8e2cf`,r.textAlign=`center`,r.textBaseline=`middle`,r.fillText(e,128,33),t){r.font=`600 18px -apple-system, system-ui, sans-serif`;let e=`✦ visiting from ${t}`,n=Math.min(244,r.measureText(e).width+24);r.fillStyle=`rgba(12,30,45,0.85)`,r.beginPath(),r.roundRect((256-n)/2,58,n,30,15),r.fill(),r.fillStyle=`#7dd3fc`,r.fillText(e,128,74)}let o=new Hi(n);o.colorSpace=at;let s=new ei(new Br({map:o,transparent:!0,depthWrite:!1}));return s.scale.set(1.7,t?.63:.42,1),s}function nu(){let e=document.createElement(`canvas`);e.width=e.height=64;let t=e.getContext(`2d`);t.font=`44px -apple-system, system-ui, sans-serif`,t.textAlign=`center`,t.textBaseline=`middle`,t.fillText(`✦`,32,34),t.fillStyle=`#ffdfae`,t.globalCompositeOperation=`source-in`,t.fillRect(0,0,64,64);let n=new Hi(e);n.colorSpace=at;let r=new ei(new Br({map:n,transparent:!0,depthWrite:!1,opacity:0}));return r.scale.set(.4,.4,1),r}var ru=class{scene;bySlug=new Map;placeById={};fetchedAtMs=0;clock=0;constructor(e){this.scene=e}sync(e,t,n){this.placeById=Object.fromEntries(t.map(e=>[e.id,e])),this.fetchedAtMs=n;let r=new Set;for(let t of e){r.add(t.slug);let e=this.bySlug.get(t.slug);if(e){e.b=t;let n=`${t.avatar?.c??1}:${t.avatar?.p??`ember`}`;n!==e.avatarKey&&(e.avatarKey=n,eu(t.avatar?.c??1,t.avatar?.p??`ember`).then(t=>{e.mat.map=t,e.mat.needsUpdate=!0}).catch(()=>{}));continue}this.add(t)}for(let[e,t]of this.bySlug)r.has(e)||(this.retire(t),this.bySlug.delete(e))}add(e){let t=Ql*(Zl[e.stage]??1),n=new An,r=new ui({transparent:!0,alphaTest:.15,side:2,depthWrite:!0,opacity:e.state===`alive`?1:.55});r.visible=!1;let i=new Si(new qi(168/224*t,t),r);i.position.y=t/2,n.add(i);let a=tu(e.name,e.kind===`visitor`?e.from||`another village`:void 0);a.position.y=t+.34,n.add(a);let o=nu();o.position.y=t+.72,n.add(o);let s={group:n,paper:i,mat:r,tag:a,mark:o,b:e,avatarKey:`${e.avatar?.c??1}:${e.avatar?.p??`ember`}`,h:t,bobPhase:0,lastXZ:null,sensedAt:-1e9,frozen:null};eu(e.avatar?.c??1,e.avatar?.p??`ember`).then(e=>{r.map=e,r.visible=!0,r.needsUpdate=!0}).catch(()=>{r.color.set(v[e.avatar?.p??`ember`]?.c1??`#c46a3f`),r.visible=!0}),this.scene.add(n),this.bySlug.set(e.slug,s)}retire(e){this.scene.remove(e.group),e.paper.geometry.dispose(),e.mat.dispose(),e.tag.material.map?.dispose(),e.tag.material.dispose(),e.mark.material.map?.dispose(),e.mark.material.dispose()}update(e,t){this.clock+=e;for(let n of this.bySlug.values()){let[r,i]=u(n.b,this.placeById,this.fetchedAtMs),a=r/Kl+16,o=i/Kl+16,s=t.x-a,c=t.z-o,l=Math.hypot(s,c)<ql,d=this.clock-n.sensedAt;l&&d>Yl&&(n.sensedAt=this.clock,n.frozen=[n.group.position.x||a,n.group.position.z||o]);let f=this.clock-n.sensedAt<Jl,p=a,m=o;if(f&&n.frozen)[p,m]=n.frozen;else if(n.frozen){let e=Math.min(1,(this.clock-n.sensedAt-Jl)/Xl);p=n.frozen[0]+(a-n.frozen[0])*e,m=n.frozen[1]+(o-n.frozen[1])*e,e>=1&&(n.frozen=null)}let h=(n.lastXZ?Math.hypot(p-n.lastXZ[0],m-n.lastXZ[1]):0)/Math.max(e,1e-6)>.02&&!f;h&&(n.bobPhase+=e*7);let g=h?Math.sin(n.bobPhase)*.055:0,_=h?Math.abs(Math.sin(n.bobPhase))*.05:0;n.lastXZ=[p,m];let v=this.clock-n.sensedAt,y=v<.45?Math.sin(v/.45*Math.PI)*.28:0,b=n.mark.material;b.opacity=f?Math.min(1,d*4)*(1-Math.max(0,d-Jl+.6)/.6):0,n.group.position.set(p,3+_+y,m),n.paper.rotation.z=g,n.group.rotation.y=Math.atan2(t.x-p,t.z-m)}}dispose(){for(let e of this.bySlug.values())this.retire(e);this.bySlug.clear()}},iu=5;function au(){let e=document.createElement(`canvas`);e.width=96,e.height=128;let t=e.getContext(`2d`);t.clearRect(0,0,96,128),t.fillStyle=`#ffffff`,t.beginPath(),t.moveTo(14,118),t.lineTo(14,56),t.arc(48,56,34,Math.PI,0),t.lineTo(82,118);for(let e=0;e<4;e++){let n=82-e*17;t.quadraticCurveTo(n-8.5,106,n-17,118)}t.closePath(),t.fill(),t.globalCompositeOperation=`destination-out`,t.beginPath(),t.arc(38,52,5,0,7),t.fill(),t.beginPath(),t.arc(58,52,5,0,7),t.fill();let n=new Hi(e);return n.colorSpace=at,n}function ou(e,t){let n=document.createElement(`canvas`);n.width=256,n.height=64;let r=n.getContext(`2d`);r.font=`600 26px -apple-system, system-ui, sans-serif`;let i=Math.min(240,r.measureText(e).width+30),a=(256-i)/2;r.fillStyle=`rgba(23,20,16,0.88)`,r.beginPath(),r.roundRect(a,12,i,40,20),r.fill(),r.lineWidth=2,r.strokeStyle=t,r.beginPath(),r.roundRect(a,12,i,40,20),r.stroke(),r.fillStyle=t,r.textAlign=`center`,r.textBaseline=`middle`,r.fillText(e,128,33);let o=new Hi(n);return o.colorSpace=at,o}var su={parent:`#c4b5fd`,visitor:`#fcd9a0`},cu={parent:12035822,visitor:15782554},lu=1.9,uu=class{scene;byId=new Map;bodyGeo=new qi(lu*.75,lu);tex=au();constructor(e){this.scene=e}toBlock(e){return[e[0]/iu+16,e[1]/iu+16]}sync(e){let t=new Set;for(let n of e){t.add(n.id);let[e,r]=this.toBlock(n.xy),i=this.byId.get(n.id);if(i){i.g=n,i.target.set(e,r);let t=n.kind===`parent`?`parent`:n.name;if(i.pill.userData.label!==t){i.pill.material.map?.dispose();let e=ou(t,su[n.kind]);i.pill.material.map=e,i.pill.material.needsUpdate=!0,i.pill.userData.label=t}continue}let a=new An,o=new ui({map:this.tex,transparent:!0,opacity:.5,color:cu[n.kind],depthWrite:!1,side:2}),s=new Si(this.bodyGeo,o);s.position.y=lu/2+.15;let c=n.kind===`parent`?`parent`:n.name,l=new ei(new Br({map:ou(c,su[n.kind]),transparent:!0,depthWrite:!1}));l.scale.set(1.7,.42,1),l.position.y=lu+.5,l.userData.label=c,a.add(s,l),a.position.set(e,3,r),this.scene.add(a),this.byId.set(n.id,{group:a,body:s,bodyMat:o,pill:l,g:n,target:new Mt(e,r),cur:new Mt(e,r),phase:0})}for(let[e,n]of this.byId)t.has(e)||(this.retire(n),this.byId.delete(e))}retire(e){this.scene.remove(e.group),e.bodyMat.dispose(),e.pill.material.map?.dispose(),e.pill.material.dispose()}update(e,t){let n=Math.min(1,e*3);for(let r of this.byId.values()){r.cur.lerp(r.target,n),r.phase+=e;let i=Math.sin(r.phase*1.5)*.08;r.group.position.set(r.cur.x,3.1+i,r.cur.y),r.body.rotation.y=Math.atan2(t.x-r.cur.x,t.z-r.cur.y)}}dispose(){for(let e of this.byId.values())this.retire(e);this.byId.clear(),this.bodyGeo.dispose(),this.tex.dispose()}},du=2.4,fu=class{scene;items=[];postGeo=new Ki(.12,1.05,.12);postMat=new ia({color:7033144});boardGeo=new Ki(.6,.42,.06);boardMat=new ia({color:15260864});glowGeo=new qi(.66,.48);glowMat=new ui({color:16770733,transparent:!0,opacity:.32,depthWrite:!1,side:2});constructor(e,t){this.scene=e;for(let e of t){let t=new An,n=new Si(this.postGeo,this.postMat);n.position.y=.52;let r=new Si(this.boardGeo,this.boardMat);r.position.set(0,1,.06),r.rotation.x=-.5;let i=new Si(this.glowGeo,this.glowMat);i.position.set(0,1,.09),i.rotation.x=-.5,t.add(n,r,i),t.position.set(e.bx+.5,3,e.bz+.5),this.scene.add(t),this.items.push({l:e,group:t})}}update(e,t){for(let{group:n}of this.items)n.position.y=3+Math.sin(e*1.6+n.position.x)*.03,n.rotation.y=Math.atan2(t.x-n.position.x,t.z-n.position.z)}nearest(e){let t=null,n=du;for(let{l:r,group:i}of this.items){let a=Math.hypot(e.x-i.position.x,e.z-i.position.z);a<n&&(n=a,t=r)}return t}dispose(){for(let{group:e}of this.items)this.scene.remove(e);this.items=[],this.postGeo.dispose(),this.postMat.dispose(),this.boardGeo.dispose(),this.boardMat.dispose(),this.glowGeo.dispose(),this.glowMat.dispose()}},pu=5,mu=2.6;function hu(){let e=document.createElement(`canvas`);e.width=64,e.height=48;let t=e.getContext(`2d`);t.fillStyle=`#f2ead2`,t.fillRect(0,0,64,48),t.strokeStyle=`rgba(120,94,58,.85)`,t.lineWidth=2,t.strokeRect(1,1,62,46),t.strokeStyle=`rgba(90,70,50,.5)`,t.lineWidth=1.5;for(let e=12;e<42;e+=8){t.beginPath(),t.moveTo(8,e);for(let n=8;n<=50-e%16;n+=6)t.lineTo(n,e+Math.sin(n*1.7+e)*1.2);t.stroke()}let n=new Hi(e);return n.colorSpace=at,n}var gu=class{scene;byId=new Map;postGeo=new Ki(.09,1,.09);postMat=new ia({color:7033144});paperGeo=new qi(.62,.46);paperMat;constructor(e){this.scene=e,this.paperMat=new ui({map:hu(),side:2})}sync(e){let t=new Set;for(let n of e){t.add(n.id);let e=this.byId.get(n.id);if(e){e.note=n;continue}let r=new An,i=new Si(this.postGeo,this.postMat);i.position.y=.5;let a=new Si(this.paperGeo,this.paperMat);a.position.y=1.06,r.add(i,a),r.position.set(n.x/pu+16,3,n.y/pu+16),this.scene.add(r),this.byId.set(n.id,{group:r,paper:a,note:n})}for(let[e,n]of this.byId)t.has(e)||(this.scene.remove(n.group),this.byId.delete(e))}update(e){for(let t of this.byId.values())t.paper.rotation.y=Math.atan2(e.x-t.group.position.x,e.z-t.group.position.z)}nearest(e){let t=null,n=mu;for(let r of this.byId.values()){let i=Math.hypot(e.x-r.group.position.x,e.z-r.group.position.z);i<n&&(n=i,t=r.note)}return t}dispose(){for(let e of this.byId.values())this.scene.remove(e.group);this.byId.clear(),this.postGeo.dispose(),this.postMat.dispose(),this.paperGeo.dispose(),this.paperMat.map?.dispose(),this.paperMat.dispose()}},_u=(e,t,n)=>e<t?t:e>n?n:e,vu=(e,t,n)=>e+(t-e)*n,yu=(e,t,n)=>{let r=_u((n-e)/(t-e),0,1);return r*r*(3-2*r)},bu=[{d:[1,0,0],c:[[1,0,1],[1,0,0],[1,1,0],[1,1,1]],sh:.74},{d:[-1,0,0],c:[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],sh:.74},{d:[0,1,0],c:[[0,1,1],[1,1,1],[1,1,0],[0,1,0]],sh:1},{d:[0,-1,0],c:[[0,0,0],[1,0,0],[1,0,1],[0,0,1]],sh:.55},{d:[0,0,1],c:[[0,0,1],[1,0,1],[1,1,1],[0,1,1]],sh:.87},{d:[0,0,-1],c:[[1,0,0],[0,0,0],[0,1,0],[1,1,0]],sh:.87}],xu=[[0,0],[1,0],[1,1],[0,1]];function Su(e,t,n){let r=[],i=[],a=[],o=[],s=[],c=[],l=[],u=[],d=[];for(let t=0;t<24;t++)for(let f=0;f<232;f++)for(let p=0;p<232;p++){let m=e.get(p,t,f);if(m===Z.AIR)continue;let h=Dl(m),g=Tl[m];for(let _=0;_<bu.length;_++){let v=bu[_],y=e.get(p+v.d[0],t+v.d[1],f+v.d[2]);if(m===Z.WATER){if(_!==2||y!==Z.AIR)continue}else if(y!==Z.AIR&&y!==Z.WATER)continue;let b=g[_===2?0:_===3?2:1],[x,S]=n(b),C=.0015;if(h){let e=c.length/3;for(let e=0;e<4;e++){let n=v.c[e];c.push(p+n[0],t+n[1],f+n[2]),l.push(v.d[0],v.d[1],v.d[2]),u.push(x+C+xu[e][0]*(Ml-2*C),S+C+xu[e][1]*(Ml-2*C))}d.push(e,e+1,e+2,e,e+2,e+3)}else{let e=r.length/3,n=m===Z.GRASS||m===Z.MEADOW?.94+(p*7+f*13)%5*.03:1;for(let e=0;e<4;e++){let s=v.c[e];r.push(p+s[0],t+s[1],f+s[2]),i.push(v.d[0],v.d[1],v.d[2]),a.push(x+C+xu[e][0]*(Ml-2*C),S+C+xu[e][1]*(Ml-2*C)),o.push(v.sh*n,v.sh*n,v.sh*n)}s.push(e,e+1,e+2,e,e+2,e+3)}}}let f=new Pr;f.setAttribute(`position`,new Sr(r,3)),f.setAttribute(`normal`,new Sr(i,3)),f.setAttribute(`uv`,new Sr(a,2)),f.setAttribute(`color`,new Sr(o,3)),f.setIndex(s);let p=new Si(f,new ia({map:t,vertexColors:!0})),m=new Pr;return m.setAttribute(`position`,new Sr(c,3)),m.setAttribute(`normal`,new Sr(l,3)),m.setAttribute(`uv`,new Sr(u,2)),m.setIndex(d),{solid:p,glow:new Si(m,new ui({map:t}))}}function Cu(e,t,n){let r=new xl({canvas:e,antialias:!1});r.setPixelRatio(Math.min(window.devicePixelRatio,2));let i=new Bn,a=new Ia(75,1,.08,600);a.rotation.order=`YXZ`,i.fog=new zn(10274024,60,220);let{texture:o,uvFor:s}=jl(),{solid:c,glow:l}=Su(t,o,s);i.add(c),i.add(l);let u=new wa(14674687,10129272,.9),d=new za(16777215,1.5);i.add(u,d,d.target);let f=new ru(i),p=new gu(i),m=new fu(i,t.lecterns),h=new uu(i),g=()=>({x:Math.min(1e3,Math.max(0,Math.round((E.pos.x-16)*5))),y:Math.min(1e3,Math.max(0,Math.round((E.pos.z-16)*5)))}),_=new An;i.add(_);let v=new Si(new qi(26,26),new ui({color:16769658,fog:!1}));v.position.set(300,0,80),_.add(v);let y=new Si(new qi(18,18),new ui({color:15265535,fog:!1}));y.position.set(-300,0,-80),_.add(y);let b=new Pr;{let e=[],t=20858,n=()=>(t=t*16807%2147483647,t/2147483647);for(let t=0;t<360;t++){let t=n()*Math.PI*2,r=Math.acos(n()*2-1);e.push(340*Math.sin(r)*Math.cos(t),340*Math.cos(r),340*Math.sin(r)*Math.sin(t))}b.setAttribute(`position`,new Sr(e,3))}let x=new Pi({color:13490431,size:1.6,sizeAttenuation:!1,transparent:!0,opacity:0,fog:!1}),S=new zi(b,x);_.add(S);let C=new ui({color:16777215,transparent:!0,opacity:.5,fog:!1}),w=new Ki(1,1,1),T=[];for(let e=0;e<14;e++){let t=new Si(w,C),n=t=>(e*2654435761+t*40503)%1e3/1e3;t.scale.set(14+n(1)*22,1.6,9+n(2)*12),t.position.set(n(3)*232,46+n(4)*10,n(5)*232),i.add(t),T.push(t)}let E={pos:{x:t.spawn.x,y:t.spawn.y,z:t.spawn.z},vel:{x:0,y:0,z:0},w:.3,h:1.8,onGround:!1},D=t.spawn.yaw,O=-.05,k=!1,A=0,j={},M=!1,N=!1,P=!1,F={x:0,y:0},I=!1,ee=!1,te=(e,n,r)=>El(t.get(e,n,r));function L(e,t){if(!t)return!1;let n=E.pos;n[e]+=t;let r=E.w,i=E.h,a=.001,o=Math.floor(n.x-r),s=Math.floor(n.x+r-1e-9),c=Math.floor(n.y),l=Math.floor(n.y+i-1e-9),u=Math.floor(n.z-r),d=Math.floor(n.z+r-1e-9);for(let f=c;f<=l;f++)for(let c=u;c<=d;c++)for(let l=o;l<=s;l++)if(te(l,f,c))return e===`x`?n.x=t>0?l-r-a:l+1+r+a:e===`y`?n.y=t>0?f-i-a:f+1+a:n.z=t>0?c-r-a:c+1+r+a,!0;return!1}function R(e){let t=Math.max(Math.abs(E.vel.x),Math.abs(E.vel.y),Math.abs(E.vel.z)),n=Math.max(1,Math.ceil(t*e/.4)),r=e/n;E.onGround=!1;for(let e=0;e<n;e++)L(`y`,E.vel.y*r)&&(E.vel.y<0&&(E.onGround=!0),E.vel.y=0),L(`x`,E.vel.x*r)&&(E.vel.x=0),L(`z`,E.vel.z*r)&&(E.vel.z=0)}function ne(e){let n=_u((j.KeyW?1:0)-(j.KeyS?1:0)+F.y,-1,1),r=_u((j.KeyD?1:0)-(j.KeyA?1:0)+F.x,-1,1),i=-Math.sin(D)*n+Math.cos(D)*r,o=-Math.cos(D)*n-Math.sin(D)*r,s=Math.hypot(i,o)||1;i/=s,o/=s;let c=Math.hypot(n,r)>.12;if(k){let t=(j.Space||I?1:0)-(j.ShiftLeft||j.ShiftRight?1:0);I=!1,E.pos.x=_u(E.pos.x+(c?i*11*e:0),1,231),E.pos.z=_u(E.pos.z+(c?o*11*e:0),1,231),E.pos.y=_u(E.pos.y+t*11*.8*e,1,38),E.vel.x=E.vel.y=E.vel.z=0}else{let n=j.ShiftLeft||j.ShiftRight?6.4:4.4,r=E.onGround?12:4,a=Math.min(1,r*e);E.vel.x+=((c?i*n:0)-E.vel.x)*a,E.vel.z+=((c?o*n:0)-E.vel.z)*a,E.vel.y-=30*e,(j.Space||I)&&E.onGround&&(E.vel.y=9.2),I=!1,E.vel.y=Math.max(E.vel.y,-38),R(e),E.pos.y<-10&&(E.pos={...t.spawn,y:5},E.vel.y=0)}let l=Math.hypot(E.vel.x,E.vel.z);E.onGround&&l>1&&(A+=e*l*1.7);let u=!k&&E.onGround&&l>1?Math.sin(A*2)*.05:0;a.position.set(E.pos.x,E.pos.y+1.62+u,E.pos.z),a.rotation.y=D,a.rotation.x=O}let re=new Ln(10274024),ie=new Ln(1053995),z=new Ln(16755566),ae=new Ln,oe=new Ln;function se(){let e=new Date,t=((e.getHours()+e.getMinutes()/60+e.getSeconds()/3600-6)/24%1+1)%1*Math.PI*2,n=Math.sin(t),r=yu(-.12,.14,n),o=_u(1-Math.abs(n)*3.2,0,1)*.6;ae.copy(ie).lerp(re,r),oe.copy(ae).lerp(z,o*(.25+r*.75)),i.background=oe,i.fog.color.copy(oe),u.intensity=.55+r*1.25,d.intensity=.18+r*2.3,d.color.setHSL(.12,.5,vu(.6,.95,1-o)),_.rotation.z=t,_.position.set(E.pos.x,E.pos.y,E.pos.z),v.lookAt(a.position),y.lookAt(a.position),x.opacity=(1-r)*.9,C.color.setScalar(vu(.3,1,r));let s=Math.cos(t),c=Math.sin(t),l=c<0;d.position.set(E.pos.x+(l?-s:s)*120,E.pos.y+Math.abs(c)*120+16,E.pos.z+40),d.target.position.set(E.pos.x,E.pos.y,E.pos.z)}let ce=``;function le(){let e=E.pos.x,r=E.pos.z,i=``,a=``,o=1/0;for(let n of t.labels){if(e>=n.x0-1&&e<=n.x1+2&&r>=n.z0-1&&r<=n.z1+2){i=`at ${n.name}`;break}let t=Math.max(n.x0-e,0,e-n.x1-1),s=Math.max(n.z0-r,0,r-n.z1-1),c=Math.hypot(t,s);c<o&&(o=c,a=n.name)}i||=t.get(Math.floor(e),Math.floor(E.pos.y)-1,Math.floor(r))===Z.PATH?`on the street`:o<10?`near ${a}`:e<8||e>224||r<8||r>224?`at the edge of the woods`:`on the open green`;let s=p.nearest({x:e,z:r}),c=m.nearest({x:e,z:r}),l=`${i}|${k}|${s?.id??``}|${c?.placeId??``}`;l!==ce&&(ce=l,n.onStatus({place:i,phase:k,note:s,readable:c}))}let B=()=>{let e=document.activeElement;return!!e&&(e.tagName===`INPUT`||e.tagName===`TEXTAREA`||e.tagName===`SELECT`||e.isContentEditable)},ue=()=>{document.pointerLockElement===e?document.exitPointerLock():N&&(M=!1,N=!1,P=!1,ge(),n.onLock(!1))},V=()=>(k=!k,E.vel.y=0,ce=``,k),de=()=>{n.onPlant&&(ue(),n.onPlant(g()))},fe=()=>{if(!n.onRead)return!1;let e=m.nearest({x:E.pos.x,z:E.pos.z});return e?(ue(),n.onRead(e),!0):!1},pe=()=>{if(!n.onPull)return;let e=p.nearest({x:E.pos.x,z:E.pos.z});e&&n.onPull(e)},me=e=>{if(!B()){if(e.code===`Space`&&e.preventDefault(),j[e.code]=!0,!M){e.code===`Escape`&&N&&(M=!1,N=!1,P=!1,ge(),n.onLock(!1));return}e.code===`KeyF`?V():e.code===`KeyE`?de():e.code===`KeyX`?pe():e.code===`KeyR`?fe():e.code===`Escape`&&N&&(M=!1,N=!1,P=!1,ge(),n.onLock(!1))}},he=e=>{B()||(j[e.code]=!1)},ge=()=>{for(let e in j)j[e]=!1;F.x=F.y=0},_e=null,ve=e=>{if(!ee||e.alpha==null||e.beta==null)return;let t=e.alpha*Math.PI/180,n=e.beta*Math.PI/180;if(!_e){_e={yaw:D,pitch:O,a:t,b:n};return}let r=t-_e.a;for(;r>Math.PI;)r-=Math.PI*2;for(;r<-Math.PI;)r+=Math.PI*2;D=_e.yaw+r,O=_u(_e.pitch-(n-_e.b),-1.45,1.45)},ye=e=>{!M||N&&!P||(D-=e.movementX*.0022,O=_u(O-e.movementY*.0022,-1.55,1.55))},be=t=>{N&&M&&t.target===e&&(P=!0)},xe=()=>{P=!1},Se=()=>{N||(M=document.pointerLockElement===e,M||ge(),n.onLock(M))};document.addEventListener(`keydown`,me),document.addEventListener(`keyup`,he),document.addEventListener(`mousemove`,ye),document.addEventListener(`mousedown`,be),document.addEventListener(`mouseup`,xe),document.addEventListener(`pointerlockchange`,Se),window.addEventListener(`blur`,ge);let Ce=()=>{let t=e.clientWidth||1,n=e.clientHeight||1;r.setSize(t,n,!1),a.aspect=t/n,a.updateProjectionMatrix()};Ce();let we=new ResizeObserver(Ce);we.observe(e);let Te=performance.now(),Ee=0,H=e=>{let t=_u((e-Te)/1e3,0,.05);Te=e,M?ne(t):a.position.set(E.pos.x,E.pos.y+1.62,E.pos.z),se(),f.update(t,E.pos),p.update(E.pos),m.update(e/1e3,E.pos),h.update(t,E.pos),Ee-=t,Ee<=0&&(Ee=.25,le());for(let e of T)e.position.x+=t*1.2,e.position.x>272&&(e.position.x=-40);r.render(i,a)};r.setAnimationLoop(H);let De=()=>{M||(N=!0,M=!0,n.onLock(!0))};return window.__fpv={player:E,get locked(){return M},get soft(){return N},get keys(){return j},get yaw(){return D},look:(e,t)=>{D+=e,O=_u(O+t,-1.55,1.55)},step:(e=100,t=1)=>{for(let n=0;n<t;n++)H(Te+e)},setMove:(e,t)=>{F.x=_u(e,-1,1),F.y=_u(t,-1,1)},jump:()=>{I=!0},get phase(){return k}},{lock:()=>{if(!M)try{let t=e.requestPointerLock();t&&typeof t.catch==`function`?t.catch(De):t||window.setTimeout(()=>{document.pointerLockElement!==e&&De()},250)}catch{De()}},setBeings:(e,t,n)=>f.sync(e,t,n),setNotes:e=>{p.sync(e),ce=``},setGhosts:e=>h.sync(e),positionUnits:g,enterTouch:()=>{M||(N=!0,M=!0,n.onLock(!0))},setMove:(e,t)=>{F.x=_u(e,-1,1),F.y=_u(t,-1,1)},look:(e,t)=>{D-=e,O=_u(O-t,-1.55,1.55)},jump:()=>{I=!0},toggleFly:()=>V(),note:()=>de(),read:()=>fe(),setGyro:e=>{ee=e,_e=null,e?window.addEventListener(`deviceorientation`,ve):window.removeEventListener(`deviceorientation`,ve)},dispose:()=>{r.setAnimationLoop(null),document.removeEventListener(`keydown`,me),document.removeEventListener(`keyup`,he),document.removeEventListener(`mousemove`,ye),document.removeEventListener(`mousedown`,be),document.removeEventListener(`mouseup`,xe),document.removeEventListener(`pointerlockchange`,Se),window.removeEventListener(`deviceorientation`,ve),window.removeEventListener(`blur`,ge),we.disconnect(),document.pointerLockElement===e&&document.exitPointerLock(),f.dispose(),p.dispose(),m.dispose(),h.dispose(),c.geometry.dispose(),c.material.dispose(),l.geometry.dispose(),l.material.dispose(),b.dispose(),x.dispose(),w.dispose(),C.dispose(),v.geometry.dispose(),v.material.dispose(),y.geometry.dispose(),y.material.dispose(),o.dispose(),r.dispose()}}}var $=r(),wu=(0,T.lazy)(()=>a(()=>import(`./BuildingReader-27wCQNj3.js`),__vite__mapDeps([0,1,2,3,4,5,6]))),Tu=(0,T.lazy)(()=>a(()=>import(`./MobileControls-ElNDDWgv.js`),__vite__mapDeps([7,1,2,3,4,5,6]))),Eu=typeof window<`u`&&(!!window.matchMedia?.(`(pointer: coarse)`).matches||/[?&]touch=1/.test(window.location.search)),Du=`rounded-xl border border-[#4a4436] bg-[#171410]/90 text-[#e8e2cf]`,Ou=`rounded-md border border-[#4a4436] bg-[#171410]/75 px-2.5 py-1 text-[11px] text-[#e8e2cf] backdrop-blur-sm`,ku=280;function Au(){let[e,t]=(0,T.useState)(``);return(0,T.useEffect)(()=>{let e=()=>t(new Date().toLocaleTimeString([],{hour:`2-digit`,minute:`2-digit`}));e();let n=window.setInterval(e,15e3);return()=>window.clearInterval(n)},[]),e}function ju({mode:e,name:t}){return e===`parent`?(0,$.jsx)(`div`,{className:`${Ou} border-violet-400/50 text-violet-200`,children:`parent`}):(0,$.jsx)(`div`,{className:`${Ou} border-amber-400/50 text-amber-200`,children:t||`visitor`})}function Mu({note:e}){return e.author_kind===`parent`?(0,$.jsx)(`span`,{className:`rounded border border-violet-400/50 px-1.5 py-0.5 text-[10px] text-violet-200`,children:`parent`}):(0,$.jsx)(`span`,{className:`rounded border border-amber-400/50 px-1.5 py-0.5 text-[10px] text-amber-200`,children:e.author})}function Nu({data:e,onClose:t,mode:n=`parent`,visitorName:r}){let i=(0,T.useRef)(null),a=(0,T.useRef)(null),[l,u]=(0,T.useState)(!1),[d,v]=(0,T.useState)(!1),[D,O]=(0,T.useState)({place:``,phase:!1,note:null,readable:null}),[k,A]=(0,T.useState)(!1),[j,M]=(0,T.useState)(null),[N,P]=(0,T.useState)(null),[F,I]=(0,T.useState)(``),[ee,te]=(0,T.useState)(!1),[L,R]=(0,T.useState)(``),[ne,re]=(0,T.useState)(!1),ie=(0,T.useRef)(0),z=(0,T.useRef)(null),ae=(0,T.useRef)(`g-${Math.random().toString(36).slice(2,10)}`),oe=Au();(0,T.useEffect)(()=>{if(!Eu)return;let e=window.matchMedia(`(orientation: portrait)`),t=()=>re(e.matches);return t(),e.addEventListener(`change`,t),()=>e.removeEventListener(`change`,t)},[]),(0,T.useEffect)(()=>{if(!Eu)return;let e=document.querySelector(`meta[name=viewport]`);if(!e)return;let t=e.getAttribute(`content`);return e.setAttribute(`content`,`width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no`),()=>{t!=null&&e.setAttribute(`content`,t)}},[]);let se=(0,T.useMemo)(()=>n===`parent`?{refetch:f,plant:(e,t,n)=>o(e,t,n),pull:e=>C(e),presence:(e,t)=>c(e,t),ghost:(e,t)=>S(ae.current,e,t),ghostLeave:()=>w(ae.current),files:e=>p(e),file:(e,t)=>h(e,t)}:{refetch:y,plant:(e,t,n)=>m(e,t,n,r||`visitor`),pull:null,presence:(e,t)=>b(e,t,r||``),ghost:(e,t)=>s(ae.current,e,t,r||``),ghostLeave:()=>_(ae.current,r||``),files:e=>g(e),file:(e,t)=>x(e,t)},[n,r]),ce=(0,T.useMemo)(()=>Gl(e),[e]),le=(0,T.useCallback)(async()=>{try{let e=await se.refetch();a.current?.setBeings(e.beings,e.places,Date.now()),a.current?.setNotes(e.notes??[])}catch{}},[se]);(0,T.useEffect)(()=>{let t=i.current;if(!t)return;let r=!0,o=Cu(t,ce,{onLock:e=>{u(e),e&&r&&(r=!1,v(!0),A(!0),ie.current=window.setTimeout(()=>A(!1),8e3))},onStatus:O,onPlant:e=>{R(``),I(``),P(e)},onPull:n===`parent`?e=>{C(e.id).then(le).catch(()=>{})}:void 0,onRead:t=>{let n=e.places.find(e=>e.id===t.placeId);n&&M(n)}});o.setBeings(e.beings,e.places,Date.now()),o.setNotes(e.notes??[]);let s=window.setInterval(()=>{se.refetch().then(e=>{a.current?.setBeings(e.beings,e.places,Date.now()),a.current?.setNotes(e.notes??[])}).catch(()=>{})},6e4),c=window.setInterval(()=>{let e=a.current;if(!e)return;let t=e.positionUnits(),n=z.current;n&&Math.hypot(t.x-n.x,t.y-n.y)<10||(z.current=t,se.presence(t.x,t.y).catch(()=>{}))},12e3),l=window.setInterval(()=>{let e=a.current;if(!e)return;let t=e.positionUnits();se.ghost(t.x,t.y).then(e=>a.current?.setGhosts(e.ghosts)).catch(()=>{})},2e3);return a.current=o,()=>{a.current=null,window.clearInterval(s),window.clearInterval(c),window.clearInterval(l),window.clearTimeout(ie.current),se.ghostLeave().catch(()=>{}),o.dispose()}},[ce,e,se,n]);let B=()=>{let e=a.current;e&&(Eu?e.enterTouch():e.lock())},ue=async()=>{if(!(!N||!F.trim())){te(!0),R(``);try{await se.plant(N.x,N.y,F.trim()),await le(),P(null),I(``),B()}catch(e){R(e instanceof Error?e.message:`the sign would not stand`)}finally{te(!1)}}},V=()=>{P(null),I(``),B()};return(0,E.createPortal)((0,$.jsxs)(`div`,{className:`fixed inset-0 z-[90] bg-[#0c0f0a]`,children:[(0,$.jsx)(`canvas`,{ref:i,className:`block h-full w-full cursor-crosshair`,onClick:B}),l&&(0,$.jsxs)(`div`,{className:`pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 mix-blend-difference`,children:[(0,$.jsx)(`div`,{className:`absolute left-1/2 top-1/2 h-4 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-[#e8e2cf]`}),(0,$.jsx)(`div`,{className:`absolute left-1/2 top-1/2 h-0.5 w-4 -translate-x-1/2 -translate-y-1/2 bg-[#e8e2cf]`})]}),Eu&&l&&a.current&&(0,$.jsx)(T.Suspense,{fallback:null,children:(0,$.jsx)(Tu,{handle:a.current,status:D})}),(0,$.jsxs)(`div`,{className:`pointer-events-none absolute left-3 top-3 flex flex-col items-start gap-1.5`,children:[D.place&&(0,$.jsx)(`div`,{className:Ou,children:D.place}),D.phase&&(0,$.jsxs)(`div`,{className:`${Ou} border-violet-400/50 text-violet-200`,children:[`👻 phase`,Eu?``:` — F to walk again`]})]}),(0,$.jsxs)(`div`,{className:`pointer-events-none absolute right-3 top-3 flex items-center gap-1.5`,children:[(0,$.jsx)(ju,{mode:n,name:r}),(0,$.jsx)(`div`,{className:Ou,children:oe})]}),l&&D.note&&(0,$.jsxs)(`div`,{className:`pointer-events-none absolute bottom-16 left-1/2 w-[min(92vw,380px)] -translate-x-1/2 p-3 ${Du}`,children:[(0,$.jsxs)(`div`,{className:`mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[#8d8571]`,children:[`a sign in the grass `,(0,$.jsx)(Mu,{note:D.note}),n===`parent`&&D.note.found>0&&(0,$.jsxs)(`span`,{className:`ml-auto normal-case tracking-normal`,children:[`found by `,D.note.found]})]}),(0,$.jsx)(`p`,{className:`text-[13px] leading-relaxed`,children:D.note.text}),n===`parent`&&(0,$.jsxs)(`p`,{className:`mt-1 text-[10px] text-[#8d8571]`,children:[(0,$.jsx)(`b`,{className:`text-[#b9b19a]`,children:`X`}),` pulls it out`]})]}),l&&D.readable&&!D.note&&(0,$.jsxs)(`div`,{className:`pointer-events-none absolute bottom-16 left-1/2 -translate-x-1/2 px-4 py-2 text-[12px] ${Du}`,children:[(0,$.jsx)(`b`,{className:`text-violet-200`,children:`R`}),` — read `,D.readable.label,` in `,D.readable.placeName]}),l&&k&&!D.note&&!D.readable&&(0,$.jsxs)(`div`,{className:`pointer-events-none absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-2 text-[12px] ${Du}`,children:[`WASD walk · Space jump · `,(0,$.jsx)(`b`,{children:`F`}),` phase · `,(0,$.jsx)(`b`,{children:`E`}),` leave a note · `,(0,$.jsx)(`b`,{children:`R`}),` read · Esc pause`]}),j&&(0,$.jsx)(T.Suspense,{fallback:null,children:(0,$.jsx)(wu,{place:j,beings:e.beings,api:se,onClose:()=>{M(null),B()}})}),N&&(0,$.jsx)(`div`,{className:`absolute inset-0 grid place-items-center bg-[#0c0f0a]/55 backdrop-blur-[2px]`,children:(0,$.jsxs)(`div`,{className:`w-[min(92vw,380px)] p-4 ${Du}`,children:[(0,$.jsxs)(`div`,{className:`mb-1 flex items-center gap-1.5 text-[13px] font-semibold`,children:[`Plant a sign here `,(0,$.jsx)(ju,{mode:n,name:r})]}),(0,$.jsx)(`p`,{className:`mb-2 text-[11px] text-[#b9b19a]`,children:`The Iskre will find it when their own feet carry them near — nothing is announced, everything is discovered.`}),(0,$.jsx)(`textarea`,{autoFocus:!0,value:F,maxLength:ku,onChange:e=>I(e.target.value),onKeyDown:e=>{e.key===`Escape`&&V()},rows:3,placeholder:`a few words, left in the grass…`,className:`w-full resize-none rounded-lg border border-[#4a4436] bg-[#0c0f0a]/70 p-2 text-[16px] text-[#e8e2cf] placeholder-[#8d8571] focus:border-violet-400/50 focus:outline-none`}),(0,$.jsxs)(`div`,{className:`mt-0.5 text-right text-[10px] text-[#8d8571]`,children:[F.length,`/`,ku]}),L&&(0,$.jsx)(`p`,{className:`mb-1 text-[11px] text-red-400`,children:L}),(0,$.jsxs)(`div`,{className:`flex gap-2`,children:[(0,$.jsx)(`button`,{onClick:()=>void ue(),disabled:ee||!F.trim(),className:`flex-1 rounded-lg border border-violet-400/40 bg-violet-500/20 px-4 py-1.5 text-[12px] font-medium text-violet-100 transition-colors hover:bg-violet-500/30 disabled:opacity-40`,children:ee?`planting…`:`Plant the sign`}),(0,$.jsx)(`button`,{onClick:V,className:`rounded-lg border border-[#4a4436] px-4 py-1.5 text-[12px] text-[#b9b19a] transition-colors hover:bg-[#2a251d]`,children:`Cancel`})]})]})}),!l&&!d&&!N&&!j&&(0,$.jsx)(`div`,{className:`absolute inset-0 grid place-items-center bg-[#0c0f0a]/55`,children:(0,$.jsxs)(`div`,{className:`w-[min(92vw,380px)] p-5 text-center ${Du}`,children:[(0,$.jsxs)(`div`,{className:`flex items-center justify-center gap-2 text-[15px] font-semibold`,children:[`The village, from inside `,(0,$.jsx)(ju,{mode:n,name:r})]}),(0,$.jsx)(`p`,{className:`mt-1.5 text-[12px] leading-relaxed text-[#b9b19a]`,children:`You walk it as a quiet ghost — the same streets, the same houses, the same hour of the day. Nobody will see you. Not exactly.`}),(0,$.jsx)(`button`,{onClick:B,className:`mt-4 w-full rounded-lg border border-violet-400/40 bg-violet-500/20 px-4 py-2 text-[13px] font-medium text-violet-100 transition-colors hover:bg-violet-500/30`,children:`Step in`}),Eu?(0,$.jsxs)(`p`,{className:`mt-3 text-[11px] leading-relaxed text-[#b9b19a]`,children:[`Left `,(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`stick`}),` to walk, drag the screen (or tilt the phone) to look. Buttons at the right:`,(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:` fly`}),`, `,(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`jump`}),`,`,(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:` note`}),`, `,(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`read`}),`.`]}):(0,$.jsxs)(`div`,{className:`mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-left text-[11px] text-[#b9b19a]`,children:[(0,$.jsxs)(`span`,{children:[(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`WASD`}),` walk`]}),(0,$.jsxs)(`span`,{children:[(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`Space`}),` jump · `,(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`Shift`}),` run`]}),(0,$.jsxs)(`span`,{children:[(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`F`}),` phase — fly through walls`]}),(0,$.jsxs)(`span`,{children:[(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`E`}),` leave a note`]}),(0,$.jsxs)(`span`,{children:[(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`R`}),` read a building's work`]}),(0,$.jsxs)(`span`,{children:[(0,$.jsx)(`b`,{className:`text-[#e8e2cf]`,children:`Esc`}),` pause`]})]}),(0,$.jsx)(`button`,{onClick:t,className:`mt-3 text-[11px] text-[#8d8571] underline-offset-2 hover:text-[#b9b19a] hover:underline`,children:`stay outside`})]})}),!l&&d&&!N&&!j&&(0,$.jsx)(`div`,{className:`absolute inset-0 grid place-items-center bg-[#0c0f0a]/55 backdrop-blur-[2px]`,children:(0,$.jsxs)(`div`,{className:`w-[min(92vw,320px)] p-5 text-center ${Du}`,children:[(0,$.jsx)(`div`,{className:`text-[14px] font-semibold`,children:`The world holds its breath`}),(0,$.jsx)(`button`,{onClick:B,className:`mt-3 w-full rounded-lg border border-violet-400/40 bg-violet-500/20 px-4 py-2 text-[13px] font-medium text-violet-100 transition-colors hover:bg-violet-500/30`,children:`Keep walking`}),(0,$.jsx)(`button`,{onClick:t,className:`mt-2 w-full rounded-lg border border-[#4a4436] px-4 py-2 text-[12px] text-[#b9b19a] transition-colors hover:bg-[#2a251d]`,children:`Leave the village`})]})}),Eu&&ne&&(0,$.jsx)(`div`,{className:`absolute inset-0 z-[105] grid place-items-center bg-[#0c0f0a]/90 px-8 text-center`,children:(0,$.jsxs)(`div`,{children:[(0,$.jsx)(`div`,{className:`mx-auto mb-4 h-10 w-16 animate-pulse rounded-md border-2 border-[#b9b19a]`}),(0,$.jsx)(`div`,{className:`text-[15px] font-semibold text-[#e8e2cf]`,children:`Turn your phone sideways`}),(0,$.jsx)(`p`,{className:`mt-1 text-[12px] text-[#b9b19a]`,children:`the village opens up in landscape`})]})})]}),document.body)}export{Nu as default};