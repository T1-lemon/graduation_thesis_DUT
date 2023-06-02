import AddAPhotoIcon from '@mui/icons-material/AddAPhoto'
import React, {
  ChangeEvent,
  useContext,
  useEffect,
  useRef,
  useState
} from 'react'

import { ProgressListener } from '../../../components/Progress'
import { DataUserProfilePicture } from '../../../interface/UsersInterface'
import { useAppDispatch } from '../../../redux/hooks'
import { updateDoctorProfilePicture } from '../../../redux/thunk/adminThunk/adminDoctorThunk'
import { DataContextAddDoctor } from '../AdminAddDoctorPage'

type Props = {
  profilePicture?: string
  idDoctor?: number
}

const AdminDoctorSettingAvatar = (props: Props) => {
  const { profilePicture, idDoctor } = props
  const { setProfilePictureDoctor } = useContext(DataContextAddDoctor)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const dispatch = useAppDispatch()

  const [errorMsg, setErrorMsg] = useState<boolean>(false)
  const [selectedFile, setSelectedFile] = useState<File | string>('')

  const [urlProfile, setUrlProfile] = useState(profilePicture)

  useEffect(() => {
    setUrlProfile(profilePicture)
  }, [profilePicture])

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const MAX_FILE_SIZE = 10240
    const files = event.currentTarget.files
    if (files && files.length > 0) {
      const file = files[0]
      const fileSizeKiloBytes = file.size / 1024

      if (fileSizeKiloBytes > MAX_FILE_SIZE) {
        setErrorMsg(true)
        return
      }

      setErrorMsg(false)
      setSelectedFile(file)

      if (!profilePicture) {
        const objectUrl = URL.createObjectURL(file)
        setUrlProfile(objectUrl)
      }
    }
  }

  const handleFileClick = () => {
    fileInputRef.current?.click()
  }

  useEffect(() => {
    const handleUpdatDoctorProfilePicture = () => {
      if (selectedFile && profilePicture) {
        const data: DataUserProfilePicture = {
          userId: idDoctor,
          profilePicture: selectedFile
        }
        const fetchApiUpdateDoctorProfilePicture = async (
          data: DataUserProfilePicture
        ) => {
          ProgressListener.emit('start')
          await dispatch(updateDoctorProfilePicture(data))
          ProgressListener.emit('stop')
          setSelectedFile('')
        }
        fetchApiUpdateDoctorProfilePicture(data)
      }

      if (selectedFile && !profilePicture) {
        setProfilePictureDoctor(selectedFile)
      }
    }

    handleUpdatDoctorProfilePicture()
  }, [selectedFile])

  return (
    <div className='setting__div--avatar'>
      <div className='avatar__img' onClick={handleFileClick}>
        <input
          type='file'
          id='avatar'
          name='avatar'
          accept='image/png, image/jpeg, image/jpg'
          max-size='1000'
          className='avatar__input'
          ref={fileInputRef}
          onChange={handleFileChange}
        />
        {urlProfile ? (
          <span className='avatar--active'>
            <img className='avatar__img--active' src={urlProfile} alt='' />
          </span>
        ) : (
          <></>
        )}
        <div className='avatar__placeholder avatar__placeholder--active'>
          <AddAPhotoIcon fontSize='medium' />
          <span className='avatar__placeholder--span'>Upload photo</span>
        </div>
      </div>
      <span className='avatar__span--notice'>
        Allowed *.jpeg, *.jpg, *.png <br /> max size of 3.1 MB
      </span>
      {errorMsg ? (
        <div className='avatar__div--alert'>
          <p className='avatar__p--alert'>Image is lager than maximum limit</p>
          <span className='avatar__span--alert'>
            Please choose a smaller image
          </span>
        </div>
      ) : (
        <></>
      )}
    </div>
  )
}

export default AdminDoctorSettingAvatar
